import copy
import os
from pathlib import Path

import torch
from datasets.fingerprint import xxhash
from pydantic import BaseModel
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.datasets import CachableTokenizeFunction
from xtuner.v1.datasets.collator import build_text_ctx_labels
from xtuner.v1.datasets.utils import tokenizer_xxhash
from xtuner.v1.utils.logger import logger

from transformers import PreTrainedTokenizer

from .constant import TS_BOS, TS_CONTEXT, TS_EOS, TS_TOKEN_ALIAS


class TimeSeriesSequenceContext(SequenceContext):
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


def ts_sft_collator(
    instances: list[list[dict]],
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
) -> list[dict]:
    ret: list[dict] = []
    instance0 = instances[0][0]
    ts_bos_token = instance0["ts_bos_token"]
    ts_eos_token = instance0["ts_eos_token"]
    ts_context_token = instance0["ts_context_token"]

    for instance in instances:
        text_seq_ctx, shifted_labels = build_text_ctx_labels(
            instance=instance,
            pack_max_length=pack_max_length,
            padding_token_idx=padding_token_idx,
            pack_to_max_length=pack_to_max_length,
        )

        ts_values = [i["time_series_signals"] for i in instance if "time_series_signals" in i]
        ts_lens = [i["ts_len"] for i in instance if "ts_len" in i]
        ts_sr = [i["ts_sr"] for i in instance if "ts_sr" in i]

        if ts_values:
            ts_lens = torch.tensor(ts_lens)
            sr = torch.tensor(ts_sr)
            time_series_signals = ts_values
        else:
            time_series_signals = None
            ts_lens = None
            sr = None

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


class TimeSeriesTokenizeFnConfig(BaseModel):
    media_root: str = ""
    max_length: int | None = None

    def build(self, tokenizer, **kwargs) -> "TimeSeriesTokenizeFunction":
        return TimeSeriesTokenizeFunction(
            tokenizer,
            media_root=self.media_root,
            max_length=self.max_length,
        )


class TimeSeriesTokenizeFunction(CachableTokenizeFunction):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        media_root: str | Path,
        max_length: int | None = None,
    ):
        self.max_length = max_length

        self.media_root = media_root

        self.ts_start_token_id = tokenizer.convert_tokens_to_ids(
            TS_BOS,
        )
        self.ts_context_token_id = tokenizer.convert_tokens_to_ids(TS_CONTEXT)
        self.ts_end_token_id = tokenizer.convert_tokens_to_ids(
            TS_EOS,
        )
        self._hash: str | None = None

        super().__init__(tokenizer)

    def __call__(self, item: dict, **kwargs):
        if self._is_time_series(item["messages"]):
            if self.state == "cache":
                return self.calc_num_tokens_time_series_get_item(item)
            else:
                return self.time_series_get_item(item)
        else:
            return self.pure_text_get_item(item)

    def hash(self) -> str:
        if self._hash is None:
            # truncate to 16 characters prevent too long cache directory
            _tokenizer_hash = tokenizer_xxhash(self.tokenizer)[:16]
            ts_token_hash = xxhash.xxh64((TS_BOS + TS_CONTEXT + TS_EOS).encode()).hexdigest()[:16]
            self._hash = f"{_tokenizer_hash}_{ts_token_hash}"

        return self._hash

    def calc_num_tokens_time_series_get_item(self, data_item) -> dict:
        _time_series_path, time_series_extra_info = self.collect_time_series_paths_and_extra(data_item["messages"])
        _time_series_path = [f"{self.media_root}/{item}" for item in _time_series_path]
        _time_series_sampling_rate = time_series_extra_info["sampling_rate"]

        transform = self.build_ts_transform()
        _, ts_len, sampling_rate = transform(_time_series_path, _time_series_sampling_rate)

        stride = torch.floor(160 / ((1 + torch.exp(-sampling_rate / 100)) ** 6))
        patch_size = stride * 2
        embed_length = (torch.ceil((ts_len - patch_size) / stride) + 1).long()
        num_ts_tokens = (embed_length // 2 + 1) // 2

        self.replace_ts_token_naive(
            data_item,
            num_ts_tokens,
            TS_BOS,
            TS_CONTEXT,
            TS_EOS,
        )
        input_ids, _ = self.naive_apply_chat_template(data_item["messages"], self.tokenizer)

        input_ids, _ = self._truncated_input_and_labels(input_ids)
        assert (torch.tensor(input_ids) == self.ts_context_token_id).sum() == num_ts_tokens, (
            "ERROR: ts tokens are truncated"
        )
        return {"num_tokens": len(input_ids)}

    def time_series_get_item(self, data_item) -> dict:
        # 结果不对
        # ts_out = self._processor.time_series_processor(self._time_series_path, self._time_series_sampling_rate)
        # ts_values = ts_out["ts_values"]
        # ts_len = ts_out["ts_lens"]
        # sampling_rate = ts_out["sampling_rate"]
        _time_series_path, time_series_extra_info = self.collect_time_series_paths_and_extra(data_item["messages"])
        _time_series_path = [f"{self.media_root}/{item}" for item in _time_series_path]
        _time_series_sampling_rate = time_series_extra_info["sampling_rate"]

        transform = self.build_ts_transform()

        ts_values, ts_len, sampling_rate = transform(_time_series_path, _time_series_sampling_rate)

        stride = torch.floor(160 / ((1 + torch.exp(-sampling_rate / 100)) ** 6))
        patch_size = stride * 2
        embed_length = (torch.ceil((ts_len - patch_size) / stride) + 1).long()
        num_ts_tokens = (embed_length // 2 + 1) // 2

        # 特殊处理
        self.replace_ts_token_naive(
            data_item,
            num_ts_tokens,
            TS_BOS,
            TS_CONTEXT,
            TS_EOS,
        )
        input_ids, labels = self.naive_apply_chat_template(data_item["messages"], self.tokenizer)

        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        assert (torch.tensor(input_ids) == self.ts_context_token_id).sum() == num_ts_tokens, (
            "ERROR: ts tokens are truncated"
        )

        ret = dict(
            input_ids=input_ids,
            labels=labels,
            time_series_signals=ts_values,  # type: ignore
            ts_len=ts_len,
            ts_sr=sampling_rate,
            num_tokens=len(input_ids),
            ts_bos_token=self.ts_start_token_id,
            ts_eos_token=self.ts_end_token_id,
            ts_context_token=self.ts_context_token_id,
        )
        return ret

    def pure_text_get_item(self, data_item: dict) -> dict:
        messages = ChatMessages(messages=data_item["messages"], tools=data_item.get("tools"))

        is_pretrain = False
        if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
            is_pretrain = True
        assert is_pretrain is False, "Text pretrain data should not be processed by this function"

        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)

        ret = dict(
            input_ids=input_ids,
            labels=labels,
            num_tokens=len(input_ids),
        )
        return ret

    def replace_ts_token_naive(
        self,
        messages: dict,
        num_ts_tokens: int,
        time_series_start_token,
        time_series_context_token,
        time_series_end_token,
    ):
        for msg in messages["messages"]:
            if msg["role"] == "pretrain":
                assert len(messages["messages"]) == 1, "pretrain message should only have one message"
            if msg["role"] == "user" or msg["role"] == "pretrain":
                content = msg["content"]
                if isinstance(content, list):
                    for c in content:
                        if c["type"] == "text":
                            text = c["text"]
                            text = text.replace("<TS_CONTEXT>", TS_TOKEN_ALIAS)
                            ts_cnt = text.count(TS_TOKEN_ALIAS)
                            # import ipdb; ipdb.set_trace()
                            for i in range(ts_cnt):
                                ts_tokens = f"{time_series_start_token}{time_series_context_token * num_ts_tokens}{time_series_end_token}"  # type: ignore
                                text = text.replace(TS_TOKEN_ALIAS, ts_tokens)
                            c["text"] = text
            if msg["role"] == "assistant":
                msg["content"] = "<think></think>\n\n" + msg["content"]

    def naive_apply_chat_template(self, messages: list[dict], tokenizer: PreTrainedTokenizer):
        labels = []
        input_ids = []

        def _tokenize_single_content(content: str, role: str):
            if role == "user":
                loss = False
                text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            elif role == "assistant":
                loss = True
                text = f"{content}<|im_end|>"
            elif role == "system":
                loss = False
                text = f"<|im_start|>system\n{content}<|im_end|>\n"
            else:
                raise RuntimeError

            input_ids = tokenizer.encode(text, add_special_tokens=False)

            if loss:
                labels = copy.deepcopy(input_ids)
            else:
                labels = [-100] * len(input_ids)

            # Just for alignment with XTuner chat template.
            # TODO: This should be removed..
            if role == "assistant":
                input_ids.append(198)
                labels.append(-100)

            return input_ids, labels

        def process_single_turn(content: str | list[dict], role: str):
            if isinstance(content, list):
                input_ids, labels = zip(*(_tokenize_single_content(i["text"], role) for i in content if "text" in i))
                input_ids = sum(input_ids, [])
                labels = sum(labels, [])
            else:
                input_ids, labels = _tokenize_single_content(content, role)

            return input_ids, labels

        input_ids, labels = zip(*(process_single_turn(message["content"], message["role"]) for message in messages))
        input_ids = sum(input_ids, [])
        labels = sum(labels, [])

        return input_ids, labels

    def _truncated_input_and_labels(self, input_ids, labels: torch.Tensor | None = None):
        if self.max_length is not None and len(input_ids) > self.max_length:
            logger.info(
                f"WARNING: input_ids length {len(input_ids)} exceeds model_max_length {self.max_length}. truncated!"
            )
            input_ids = input_ids[: self.max_length]
            if labels is not None:
                labels = labels[: self.max_length]
        return input_ids, labels

    def collect_time_series_paths_and_extra(self, messages: list[dict]):
        time_series_paths = []
        sampling_rate_list = []
        for msg in messages:
            if msg["role"] == "user" or msg["role"] == "pretrain":
                content = msg["content"]
                if isinstance(content, list):
                    for c in content:
                        if c["type"] == "time_series_url":
                            time_series_paths.append(c["time_series_url"]["url"])
                            if "sampling_rate" in c["time_series_url"]:
                                sampling_rate = c["time_series_url"]["sampling_rate"]
                            else:
                                sampling_rate = None
                            sampling_rate_list.append(sampling_rate)

        if len(time_series_paths) > 0:
            assert len(time_series_paths) == len(sampling_rate_list) == 1

        return (
            time_series_paths,
            {"sampling_rate": sampling_rate_list},
        )

    def build_ts_transform(self, do_normalize=True, do_truncate=True, max_len=240000):
        def transform(ts_path, sr: str):
            assert len(ts_path) == 1, "Currently only one ts signal is supported."
            ts_path = ts_path[0]
            ext = os.path.splitext(ts_path)[-1].lower()
            try:
                if ext in [".wav", ".mp3", ".flac"]:
                    try:
                        import librosa
                    except ImportError:
                        raise ImportError("Please install librosa to process audio files.")
                    ts_input, sr = librosa.load(ts_path, sr=None)
                    ts_input = ts_input[:, None]  # [T, 1]
                elif ext == ".csv":
                    try:
                        import pandas as pd
                    except ImportError:
                        raise ImportError("Please install pandas to process CSV files.")
                    df = pd.read_csv(ts_path, header=None)
                    ts_input = df.values  # [T, C]
                elif ext == ".npy":
                    try:
                        import numpy as np
                    except ImportError:
                        raise ImportError("Please install numpy to process NPY files.")
                    ts_input = np.load(ts_path)  # [T, C]
                else:
                    raise ValueError(f"Unsupported file format: {ext}")

                ts_tensor = torch.from_numpy(ts_input)

                if do_normalize:
                    mean = ts_tensor.mean(dim=0, keepdim=True)
                    std = ts_tensor.std(dim=0, keepdim=True)
                    ts_tensor = (ts_tensor - mean) / (std + 1e-8)

                ts_tensor = ts_tensor.to(torch.bfloat16)

                if do_truncate:
                    if len(ts_tensor) > 240000:  # truncate to 240k to avoid oom
                        ts_tensor = ts_tensor[:240000, :]

                if len(ts_tensor.size()) == 1:
                    ts_tensor = ts_tensor.unsqueeze(-1)

                ts_len = ts_tensor.size(0)
                if sr is None or sr == 0:  # if no sr provided
                    sr = ts_len / 4
                else:
                    sr = sr[0]  # remove list

                return ts_tensor, torch.tensor(ts_len), torch.tensor(sr)

            except Exception as e:
                print(f"Error processing time series file {ts_path}: {e}")
                return None

        return transform

    def _is_time_series(self, messages: list):
        assert isinstance(messages, list)
        for single_turn in messages:
            content_list = single_turn["content"]
            if not isinstance(content_list, list):
                continue

            for content in content_list:
                if content["type"] == "time_series_url":
                    return True
        else:
            return False
