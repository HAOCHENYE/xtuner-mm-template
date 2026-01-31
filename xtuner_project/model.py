import json
from pathlib import Path

import torch
import torch.distributed as dist
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.base import BaseModel as XTunerBaseModel
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import XTunerBaseModelConfig
from xtuner.v1.model.moe.moe import MoEModelOutputs, SequenceContext
from xtuner.v1.model.utils.misc import update_weight_map_from_safetensors_index
from xtuner.v1.utils import get_logger, profile_time_and_memory

from transformers import AutoConfig, AutoModel, PreTrainedModel

from .data import TimeSeriesSequenceContext


logger = get_logger()


class TsModelConfig(XTunerBaseModelConfig):
    text_config: TransformerConfig
    ts_model_path: str | Path

    def build(self):
        return TSModel(self)

    @property
    def hf_config(self):
        return None


class TSModel(XTunerBaseModel):
    def __init__(self, config: TsModelConfig):
        super().__init__(config)
        self.language_model = config.text_config.build()
        self.language_model.requires_grad_(False)

        with torch.device("cuda"):
            ts_config = AutoConfig.from_pretrained(config.ts_model_path, trust_remote_code=True).ts_config
            self.ts_model: PreTrainedModel = AutoModel.from_pretrained(
                config.ts_model_path,
                config=ts_config,
                trust_remote_code=True,
                key_mapping={r"^model.time_series.": "model."},
            )

    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple[set[str], set[str], set[str]]:
        self.language_model.from_hf(hf_path, strict)
        return set(), set(), set()

    def save_hf(
        self,
        hf_dir: Path | str,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
    ):
        with profile_time_and_memory(f"Saving model to {hf_dir}"):
            self.language_model.save_hf(hf_dir, save_dtype, "HF")
            hf_dir = Path(hf_dir)

            weight_map_dict: dict = {}
            update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)

            # XTuner's internal save_hf implementation can efficiently save the language_model to HuggingFace format
            # under sharded states. However, since HF models haven't implemented the corresponding logic,
            # it's necessary to first gather all parameters before executing save_pretrained.
            state_dict = self._collect_full_state_dict(self.ts_model)
            if dist.get_rank() == 0:
                remap_state_dict = {f"model.time_series.{k}": v for k, v in state_dict.items()}
                self.ts_model.save_pretrained(
                    str(hf_dir),
                    state_dict=remap_state_dict,
                )
                update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)

                with open(hf_dir / "model.safetensors.index.json", "w") as f:
                    json.dump({"weight_map": weight_map_dict, "metadata": {}}, f, indent=2)

            dist.barrier()
        return (set(self.state_dict()), set(), set())

    def forward(self, seq_ctx: TimeSeriesSequenceContext, loss_ctx: CELossContext) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids

        inputs_embeds = self.language_model.embed_tokens(input_ids)  # type: ignore
        time_series_signals = seq_ctx.time_series_signals

        if time_series_signals is not None:
            ts_features, ts_pad_mask = self.get_ts_feature(
                time_series_signals, seq_ctx.ts_lens, seq_ctx.ts_sr
            )  # [B, T, C], [B, T]
            ts_features = ts_features[~ts_pad_mask].to(
                inputs_embeds.device, inputs_embeds.dtype
            )  # [num_valid_ts_tokens, C]
            B, N, C = inputs_embeds.shape
            input_ids = input_ids.reshape(B * N)
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            # replace ts_token in inputs_embeds and attention_mask
            ts_placeholder = input_ids == seq_ctx.ts_context_token
            n_ts_placeholders = ts_placeholder.sum().item()
            n_ts_tokens = ts_features.size(0)
            assert n_ts_placeholders == n_ts_tokens, (
                f"[ERROR]: Mismatch: <TS_CONTEXT> tokens={n_ts_placeholders}, ts_embeds_valid={n_ts_tokens}"
            )

            try:
                inputs_embeds[ts_placeholder] = inputs_embeds[ts_placeholder] * 0.0 + ts_features
            except Exception as e:
                logger.error(
                    f"warning: {e}, inputs_embeds[selected].shape={inputs_embeds[ts_placeholder].shape}, ts_embeds_valid.shape={ts_features.shape}"
                )
                inputs_embeds = inputs_embeds * 0.0 + ts_features.sum() * 0.0

            inputs_embeds = inputs_embeds.reshape(B, N, C)
        else:
            fake_time_series_signals = torch.zeros((1, 147, 3), device=input_ids.device, dtype=inputs_embeds.dtype)
            fake_ts_lens = torch.tensor([147], device=input_ids.device)
            fake_sr = torch.tensor([36], device=input_ids.device)
            ts_features, _ = self.get_ts_feature(fake_time_series_signals, fake_ts_lens, fake_sr)
            inputs_embeds = inputs_embeds * 0.0 + ts_features.sum() * 0.0

        lang_seq_ctx = SequenceContext(
            input_ids=None,
            cu_seq_lens_q=seq_ctx.cu_seq_lens_q,
            cu_seq_lens_k=seq_ctx.cu_seq_lens_k,
            max_length_q=seq_ctx.max_length_q,
            max_length_k=seq_ctx.max_length_k,
            position_ids=seq_ctx.position_ids,
            num_padding=seq_ctx.num_padding,
            sequence_parallel_mesh=seq_ctx.sequence_parallel_mesh,
            inputs_embeds=inputs_embeds,
        )
        outputs = self.language_model(lang_seq_ctx, loss_ctx)
        return outputs

    def get_ts_feature(self, ts_values, ts_lens, sr):
        ts_values = [i.cuda() for i in ts_values]
        ts_embeds, ts_pad_mask = self.ts_model(time_series_signals=ts_values, ts_lens=ts_lens, sr=sr)
        return ts_embeds, ts_pad_mask
