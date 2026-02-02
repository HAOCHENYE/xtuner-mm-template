# XTuner Playground - Huggingface Encoder + XTuner LLM Best Practice

This repository provides best practices for training multimodal models using Huggingface Encoder with XTuner LLM. With the rapid development of the multimodal field and the emergence of various encoders and decoders, this project demonstrates how to directly use Huggingface encoders for multimodal model training, using time series data as an example.

The XTuner training workflow revolves around three core components: **Data**, **Model**, and **Config**. This document introduces the implementation methods for these three components in sequence.

## Table of Contents

- [Quick Start](#quick-start)
- [Data Component](#data-component)
  - [Data Format](#data-format)
  - [TokenizeFunction Implementation](#tokenizefunction-implementation)
  - [CollateFunction Implementation](#collatefunction-implementation)
  - [SequenceContext Extension](#sequencecontext-extension)
- [Model Component](#model-component)
  - [Model Configuration](#model-configuration)
  - [Model Implementation](#model-implementation)
  - [Key Interfaces](#key-interfaces)
- [Training Configuration](#training-configuration)
- [Launch Training](#launch-training)

______________________________________________________________________

## Quick Start

1. Prepare data files (jsonl format)
2. Implement custom data processing functions
3. Implement custom model classes
4. Configure training parameters
5. Launch training

______________________________________________________________________

## Data Component

The XTuner data processing pipeline is: **JsonlDataset -> tokenize_fn -> collate_fn** (with a `PackingDataset` layer in between, not detailed here). `JsonlDataset` reads data from jsonl files, `tokenize_fn` parses each data item and performs preprocessing, and `collate_fn` assembles multiple data items into batches.

### Data Format

Data files use jsonl format, with each line being a JSON object. `JsonlDataset` only handles data reading, while data parsing is handled by `tokenize_fn`. Using time series signal data as an example, multimodal information needs to be included in `messages`.

**Time Series Data Example:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "time_series_url",
          "time_series_url": {
            "url": "data/example.wav",
            "sampling_rate": 100
          }
        },
        {
          "type": "text",
          "text": "Please analyze this time series signal. <TS_CONTEXT>"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "This is a normal ECG signal."
    }
  ]
}
```

**Pure Text Data Example:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, please introduce yourself."
    },
    {
      "role": "assistant",
      "content": "Hello! I am an AI assistant."
    }
  ]
}
```

**Key Points:**

- The `messages` field contains multiple rounds of conversation
- For multimodal data, `content` should be a list containing different types of content
- Time series data is specified through `time_series_url` type, including file path and sampling rate
- Use `<TS_CONTEXT>` placeholder in text to mark the insertion position of time series features

### TokenizeFunction Implementation

`TokenizeFunction` is responsible for parsing each data item read by `JsonlDataset` and converting it into the format required by `collate_fn`.

**Core Responsibilities:**

1. Determine data type (pure text vs multimodal)
2. Load and process multimodal data (e.g., time series signals)
3. Apply chat template to generate token IDs and labels
4. Calculate multimodal feature token count and replace placeholders in text

**Implementation Example:**

```python
from xtuner.v1.datasets import CachableTokenizeFunction
from transformers import PreTrainedTokenizer

class TimeSeriesTokenizeFunction(CachableTokenizeFunction):
    """TokenizeFunction for time series data

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer
        media_root (str | Path): Root directory for multimodal files
        max_length (int | None): Maximum sequence length, excess will be truncated
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        media_root: str | Path,
        max_length: int | None = None,
    ):
        self.max_length = max_length
        self.media_root = media_root

        # Get special token IDs
        self.ts_start_token_id = tokenizer.convert_tokens_to_ids("<|ts|>")
        self.ts_context_token_id = tokenizer.convert_tokens_to_ids("<TS_CONTEXT>")
        self.ts_end_token_id = tokenizer.convert_tokens_to_ids("<|/ts|>")

        super().__init__(tokenizer)

    def __call__(self, item: dict, **kwargs):
        """Process each line of data parsed by JsonlDataset"""
        if self._is_time_series(item["messages"]):
            return self.time_series_get_item(item)
        else:
            return self.pure_text_get_item(item)
```

**Key Method Descriptions:**

1. **`_is_time_series()`**: Determine if data contains time series signals
2. **`collect_time_series_paths_and_extra()`**: Extract time series file paths and metadata from messages
3. **`build_ts_transform()`**: Build time series data loading and preprocessing functions, supporting multiple formats (.wav, .mp3, .csv, .npy)
4. **`replace_ts_token_naive()`**: Replace `<TS_CONTEXT>` placeholder in text with actual token sequence (`<|ts|><TS_CONTEXT>...<TS_CONTEXT><|/ts|>`)
5. **`time_series_get_item()`**: Process multimodal data and return a dictionary containing time series features
6. **`pure_text_get_item()`**: Process pure text data

**Return Format:**

`tokenize_fn` must return a dictionary containing `input_ids` and `labels` fields. Additional multimodal data fields will be collected in `collate_fn`.

```python
{
    "input_ids": [...],              # token IDs (required)
    "labels": [...],                 # training labels (required)
    "time_series_signals": tensor,   # time series signals (multimodal only)
    "ts_len": int,                   # signal length (multimodal only)
    "ts_sr": int,                    # sampling rate (multimodal only)
    "num_tokens": int,               # total token count
    "ts_bos_token": int,             # special token IDs
    "ts_eos_token": int,
    "ts_context_token": int,
}
```

**Chat Template Processing:**

Users need to implement their own chat template application logic. In this example, the `naive_apply_chat_template()` method is used to convert multi-turn conversations into model input format:

```python
def naive_apply_chat_template(self, messages: list[dict], tokenizer):
    """Apply chat template

    Convert conversation data to model-required format:
    <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>
    """
    # Implement chat template logic
    # Return input_ids and labels
```

### CollateFunction Implementation

`collate_fn` corresponds to PyTorch DataLoader's collate function, responsible for assembling a batch of data. It mainly accomplishes two things:

1. **Text data batch aggregation**: Pack text data from multiple samples into a batch
2. **Multimodal data aggregation**: Collect all multimodal data in the batch

**Implementation Example:**

```python
from xtuner.v1.datasets.collator import build_text_ctx_labels

def ts_sft_collator(
    instances: list[list[dict]],
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
) -> list[dict]:
    """Collate function for time series data

    Args:
        instances (list[list[dict]]): Batch data, each element is a packed data item composed of multiple tokenize_fn outputs
        pack_max_length (int): Maximum packing length
        padding_token_idx (int): Padding token ID
        pack_to_max_length (bool): Whether to pack to maximum length

    Returns:
        list[dict]: Aggregated batch data, each element contains seq_ctx and shifted_labels
    """
    ret: list[dict] = []
    instance0 = instances[0][0]
    ts_bos_token = instance0["ts_bos_token"]
    ts_eos_token = instance0["ts_eos_token"]
    ts_context_token = instance0["ts_context_token"]

    for instance in instances:
        # Build text SequenceContext and labels
        text_seq_ctx, shifted_labels = build_text_ctx_labels(
            instance=instance,
            pack_max_length=pack_max_length,
            padding_token_idx=padding_token_idx,
            pack_to_max_length=pack_to_max_length,
        )

        # Collect time series data
        ts_values = [
            i["time_series_signals"] for i in instance if "time_series_signals" in i
        ]
        ts_lens = [i["ts_len"] for i in instance if "ts_len" in i]
        ts_sr = [i["ts_sr"] for i in instance if "ts_sr" in i]

        # Process time series data
        if ts_values:
            ts_lens = torch.tensor(ts_lens)
            sr = torch.tensor(ts_sr)
            time_series_signals = ts_values
        else:
            time_series_signals = None
            ts_lens = None
            sr = None

        # Build extended SequenceContext
        seq_ctx = TimeSeriesSequenceContext(
            **text_seq_ctx.data,
            ts_bos_token=ts_bos_token,
            ts_eos_token=ts_eos_token,
            ts_context_token=ts_context_token,
            time_series_signals=time_series_signals,
            ts_lens=ts_lens,
            ts_sr=sr,
        )

        ret.append(
            {
                "seq_ctx": seq_ctx,
                "shifted_labels": shifted_labels,
            }
        )

    return ret
```

**Core Flow:**

1. Use XTuner's `build_text_ctx_labels()` to handle text batch aggregation (prerequisite: `tokenize_fn` returns objects containing `input_ids` and `labels` fields)
2. Collect multimodal data from all samples in the batch (e.g., time series signals, lengths, sampling rates)
3. Build extended `TimeSeriesSequenceContext` to integrate text and multimodal data
4. Return a dictionary containing `seq_ctx` and `shifted_labels`

### SequenceContext Extension

`SequenceContext` is the data type accepted by XTuner models. In `collate_fn`, the text `SequenceContext` needs to be integrated with multimodal data into an extended `SequenceContext`.

**Implementation Example:**

```python
from xtuner.v1.data_proto import SequenceContext
import torch

class TimeSeriesSequenceContext(SequenceContext):
    """SequenceContext extension for time series data

    Args:
        time_series_signals (list[torch.FloatTensor] | torch.FloatTensor | None): Time series signal data
        ts_lens (torch.Tensor | None): Time series signal lengths
        ts_sr (torch.Tensor | None): Time series signal sampling rates
        ts_bos_token (int): Time series data start token ID
        ts_eos_token (int): Time series data end token ID
        ts_context_token (int): Time series feature placeholder token ID
    """
    time_series_signals: list[torch.FloatTensor] | torch.FloatTensor | None
    ts_lens: torch.Tensor | None
    ts_sr: torch.Tensor | None
    ts_bos_token: int
    ts_eos_token: int
    ts_context_token: int

    def __init__(
        self,
        *,
        time_series_signals: list[torch.FloatTensor] | torch.FloatTensor | None = None,
        ts_lens: torch.Tensor | None = None,
        ts_sr: torch.Tensor | None = None,
        ts_bos_token: int,
        ts_eos_token: int,
        ts_context_token: int,
        **data,
    ):
        super().__init__(**data)
        self.time_series_signals = time_series_signals
        self.ts_lens = ts_lens
        self.ts_sr = ts_sr
        self.ts_bos_token = ts_bos_token
        self.ts_eos_token = ts_eos_token
        self.ts_context_token = ts_context_token
```

**Design Points:**

- Inherit from `SequenceContext` base class
- Add modality-specific fields (e.g., time series signals, lengths, sampling rates)
- Add special token IDs (to mark modality data boundaries and placeholders)
- Pass text-related fields required by base class through `**data`

______________________________________________________________________

## Model Component

The model component requires implementing two classes: `ModelConfig` and `Model`.

### Model Configuration

Inherit from `XTunerBaseModelConfig` to define model configuration parameters.

**Implementation Example:**

```python
from xtuner.v1.model.compose.base import XTunerBaseModelConfig
from xtuner.v1.model.base import TransformerConfig

class TsModelConfig(XTunerBaseModelConfig):
    """Time series model configuration

    Args:
        text_config (TransformerConfig): Text model configuration
        ts_model_path (str | Path): Time series encoder path
    """
    text_config: TransformerConfig
    ts_model_path: str | Path

    def build(self):
        """Build model instance"""
        return TSModel(self)

    @property
    def hf_config(self):
        """Huggingface configuration conversion

        For models using trust_remote_code:
            - Return None, XTuner will automatically copy configuration files from the original directory

        For models built into Transformers repository:
            - Implement conversion from XTuner Config to Transformers Config
        """
        return None
```

**Key Points:**

1. **`build()` method**: Used to build model instances
2. **`hf_config` property**:
   - For models using `trust_remote_code` (like this example), return `None` and XTuner will automatically copy configuration files from the original directory to the save directory
   - For models built into the Transformers repository, implement conversion from XTuner Config to Transformers Config

### Model Implementation

Inherit from `XTunerBaseModel` to implement model forward propagation and save/load logic.

**Implementation Example:**

```python
from xtuner.v1.model.base import BaseModel as XTunerBaseModel
from transformers import AutoModel, AutoConfig

class TSModel(XTunerBaseModel):
    """Time series multimodal model

    Args:
        config (TsModelConfig): Model configuration
    """
    def __init__(self, config: TsModelConfig):
        super().__init__(config)

        # Load language model
        self.language_model = config.text_config.build()
        self.language_model.requires_grad_(False)  # Freeze LLM

        # [KEY] Must initialize model in cuda device context
        with torch.device("cuda"):
            ts_config = AutoConfig.from_pretrained(
                config.ts_model_path, trust_remote_code=True
            ).ts_config
            self.ts_model: PreTrainedModel = AutoModel.from_pretrained(
                config.ts_model_path,
                config=ts_config,
                trust_remote_code=True,
                key_mapping={r"^model.time_series.": "model."},
            )
```

**Key Points:**

1. **Must initialize model in `__init__` using `cuda` device context** (very important)
2. Need to implement `forward()` method
3. Need to implement `from_hf()` and `save_hf()` methods for model loading and saving

### Key Interfaces

#### 1. `forward()` Method

Implement model forward propagation logic. Refer to the implementation in the code template.

**Core Logic:**

1. Use `embed_tokens()` to get text token embeddings
2. Use multimodal encoder to extract features
3. Replace multimodal features into placeholder token positions in embeddings
4. Pass fused embeddings to the language model

**Implementation Example:**

```python
def forward(
    self, seq_ctx: TimeSeriesSequenceContext, loss_ctx: CELossContext
) -> MoEModelOutputs:
    """Forward propagation

    Args:
        seq_ctx (TimeSeriesSequenceContext): Input sequence context
        loss_ctx (CELossContext): Loss function context

    Returns:
        MoEModelOutputs: Model outputs
    """
    input_ids = seq_ctx.input_ids

    # 1. Get text embeddings
    inputs_embeds = self.language_model.embed_tokens(input_ids)

    # 2. Process time series data
    time_series_signals = seq_ctx.time_series_signals
    if time_series_signals is not None:
        # 2.1 Use time series encoder to extract features
        ts_features, ts_pad_mask = self.get_ts_feature(
            time_series_signals, seq_ctx.ts_lens, seq_ctx.ts_sr
        )
        ts_features = ts_features[~ts_pad_mask].to(
            inputs_embeds.device, inputs_embeds.dtype
        )

        # 2.2 Replace time series features into placeholder positions in embeddings
        B, N, C = inputs_embeds.shape
        input_ids = input_ids.reshape(B * N)
        inputs_embeds = inputs_embeds.reshape(B * N, C)

        ts_placeholder = input_ids == seq_ctx.ts_context_token
        inputs_embeds[ts_placeholder] = ts_features
        inputs_embeds = inputs_embeds.reshape(B, N, C)
    else:
        # Pure text data: use fake time series data to maintain gradient graph consistency
        fake_time_series_signals = torch.zeros(
            (1, 147, 3), device=input_ids.device, dtype=inputs_embeds.dtype
        )
        fake_ts_lens = torch.tensor([147], device=input_ids.device)
        fake_sr = torch.tensor([36], device=input_ids.device)
        ts_features, _ = self.get_ts_feature(
            fake_time_series_signals, fake_ts_lens, fake_sr
        )
        inputs_embeds = inputs_embeds * 0.0 + ts_features.sum() * 0.0

    # 3. Build language model input
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

    # 4. Call language model
    outputs = self.language_model(lang_seq_ctx, loss_ctx)
    return outputs
```

#### 2. `from_hf()` Method

Load model weights from Huggingface checkpoint. Refer to the implementation in the code template.

**Implementation Example:**

```python
def from_hf(
    self, hf_path: str | Path, strict: bool = True
) -> tuple[set[str], set[str], set[str]]:
    """Load weights from HF checkpoint

    Args:
        hf_path (str | Path): Checkpoint path
        strict (bool): Whether to strictly match weights

    Returns:
        tuple: (missing_keys, unexpected_keys, error_keys)
    """
    self.language_model.from_hf(hf_path, strict)
    return set(), set(), set()
```

#### 3. `save_hf()` Method

Save model in Huggingface format. Refer to the implementation in the code template.

**Implementation Example:**

```python
def save_hf(
    self,
    hf_dir: Path | str,
    save_dtype: torch.dtype = torch.bfloat16,
    safetensors_prefix: str = "model",
):
    """Save model to HF format

    Args:
        hf_dir (Path | str): Save directory
        save_dtype (torch.dtype): Save data type
        safetensors_prefix (str): Safetensors file prefix
    """
    self.unshard()
    hf_dir = Path(hf_dir)

    # 1. Save language model
    self.language_model.save_hf(hf_dir, save_dtype, "HF")

    # 2. Save time series encoder
    weight_map_dict: dict = {}
    update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)

    state_dict = self._collect_full_state_dict(self.ts_model)
    if dist.get_rank() == 0:
        remap_state_dict = {
            f"model.time_series.{k}": v for k, v in state_dict.items()
        }
        self.ts_model.save_pretrained(
            str(hf_dir),
            state_dict=remap_state_dict,
        )
        update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)

        # 3. Update weight_map
        with open(hf_dir / "model.safetensors.index.json", "w") as f:
            json.dump(
                {"weight_map": weight_map_dict, "metadata": {}}, f, indent=2
            )

    dist.barrier()
    return (set(self.state_dict()), set(), set())
```

**Save Logic:**

1. Save language model weights
2. Collect and remap time series encoder weights (add `model.time_series.` prefix)
3. Update weight_map in `model.safetensors.index.json`

______________________________________________________________________

## Training Configuration

Configure training parameters based on the implemented data and model components.

**Configuration Example (xtuner_project/config.py):**

```python
from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner_project.model import TsModelConfig
from xtuner_project.data import TimeSeriesTokenizeFnConfig

# Path configuration
model_path = "/path/to/pretrained/model"
work_dir = "/path/to/work_dir"
meta_data_path = "/path/to/data.jsonl"

# Training hyperparameters
sample_max_length = 8192
pack_max_length = 8192
global_batch_size = 8
total_epoch = 1
lr = 2e-5
lr_min = 2e-6
weight_decay = 0.05
warmup_ratio = 0.03

# Model configuration
model_cfg = TsModelConfig(
    ts_model_path=model_path,
    text_config=Qwen3MoE30BA3Config(
        max_position_embeddings=32768,
        n_routed_experts=512,
        hf_key_mapping={r"^model.": "model.language_model."},
    ),
)

# Data configuration
dataset_config = [
    {
        "dataset": DatasetConfig(
            name="time_series_dataset",
            anno_path=meta_data_path,
            media_root="/path/to/media/",
            sample_ratio=100,
            enable_sequential_sampler=True,
            class_name="VLMJsonlDataset",
        ),
        "tokenize_fn": TimeSeriesTokenizeFnConfig(
            media_root="/path/to/media/",
            max_length=sample_max_length,
        ),
    }
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    collator="xtuner_project.data.ts_sft_collator",
    num_workers=0,
    pack_extra_buffer_size=20,
)

# Optimizer and learning rate configuration
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)

# FSDP configuration
fsdp_cfg = FSDPConfig(
    recompute_ratio=1.0,
    ep_size=1,
    torch_compile=False,
)

# Trainer configuration
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=ResumeConfig(auto_resume=False),
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="square"),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=10,
    checkpoint_interval=500,
    checkpoint_maxkeep=5,
    hf_max_keep=5,
    work_dir=work_dir,
    strict_load=False,
)
```

**Configuration Points:**

1. **Model Configuration**:

   - Specify time series encoder path (`ts_model_path`)
   - Configure language model (`text_config`)
   - Set weight mapping rules (`hf_key_mapping`)

2. **Data Configuration**:

   - Use `VLMJsonlDataset` to read jsonl data
   - Specify custom `tokenize_fn` configuration class
   - Specify custom `collator` function path

3. **Training Configuration**:

   - Optimizer: AdamW
   - Learning rate scheduler: cosine with warmup
   - FSDP: Parallel configuration
   - Loss function: Cross entropy (chunk mode, saves memory)

______________________________________________________________________

## Launch Training

Launch training using XTuner's training script:

```bash
# Single GPU training
torchrun --master-port 29501 --nproc-per-node 8 -m xtuner.v1.train.cli.sft --config xtuner_project/config.py
```

______________________________________________________________________

## Summary

This repository demonstrates how to combine Huggingface Encoder with XTuner LLM for multimodal training:

1. **Data Level**: Extend `SequenceContext`, implement custom `TokenizeFunction` and `CollateFunction`
2. **Model Level**: Implement `ModelConfig` and `Model`, focusing on `forward()`, `save_hf()`, and `from_hf()` methods
3. **Configuration Level**: Integrate data and model configurations, set training hyperparameters

Users can adapt other types of Huggingface Encoders (such as vision encoders, audio encoders, etc.) by referring to this practice, quickly building their own multimodal training pipeline.

______________________________________________________________________
