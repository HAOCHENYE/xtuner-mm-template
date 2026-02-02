# XTuner Playground - Huggingface Encoder + XTuner LLM 最佳实践

本仓库提供了使用 Huggingface Encoder 与 XTuner LLM 进行多模态模型训练的最佳实践。随着多模态领域的快速发展，各种 encoder 和 decoder 层出不穷，本项目以时序数据（Time Series）为例，展示如何直接使用 Huggingface 的 encoder 进行多模态模型训练。

XTuner 的训练流程围绕三个核心组件展开：**数据（Data）**、**模型（Model）**、**配置（Config）**。本文档将依次介绍这三个组件的实现方法。

## 目录

- [快速开始](#快速开始)
- [数据组件](#数据组件)
  - [数据格式](#数据格式)
  - [TokenizeFunction 实现](#tokenizefunction-实现)
  - [CollateFunction 实现](#collatefunction-实现)
  - [SequenceContext 扩展](#sequencecontext-扩展)
- [模型组件](#模型组件)
  - [模型配置](#模型配置)
  - [模型实现](#模型实现)
  - [关键接口](#关键接口)
- [训练配置](#训练配置)
- [启动训练](#启动训练)

______________________________________________________________________

## 快速开始

1. 准备数据文件（jsonl 格式）
2. 实现自定义的数据处理函数
3. 实现自定义的模型类
4. 配置训练参数
5. 启动训练

______________________________________________________________________

## 数据组件

XTuner 的数据处理流程为：**JsonlDataset -> tokenize_fn -> collate_fn**。(中间还有一层 `PackingDataset`，此处不详细展开)其中 `JsonlDataset` 负责从 jsonl 文件中读取数据，`tokenize_fn` 负责解析每一条数据并进行预处理，`collate_fn` 负责将多条数据组装成 batch。

### 数据格式

数据文件采用 jsonl 格式，每行为一个 JSON 对象。`JsonlDataset` 只负责数据的读取，数据的解析由 `tokenize_fn` 负责。以时序信号数据为例，需要在 `messages` 中包含多模态信息。

**时序数据示例：**

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
          "text": "请分析这段时序信号。<TS_CONTEXT>"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "这是一段正常的心电信号。"
    }
  ]
}
```

**纯文本数据示例：**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下自己。"
    },
    {
      "role": "assistant",
      "content": "你好！我是一个AI助手。"
    }
  ]
}
```

**关键点：**

- `messages` 字段包含对话的多轮内容
- 对于多模态数据，`content` 需要是一个列表，包含不同类型的内容
- 时序数据通过 `time_series_url` 类型指定，包含文件路径和采样率
- 文本中使用 `<TS_CONTEXT>` 占位符标记时序特征的插入位置

### TokenizeFunction 实现

`TokenizeFunction` 负责解析 `JsonlDataset` 读取的每一条数据，将其转换成 `collate_fn` 需要的格式。

**核心职责：**

1. 判断数据类型（纯文本 vs 多模态）
2. 加载和处理多模态数据（如时序信号）
3. 应用对话模板，生成 token IDs 和 labels
4. 计算多模态特征的 token 数量并替换文本中的占位符

**实现示例：**

```python
from xtuner.v1.datasets import CachableTokenizeFunction
from transformers import PreTrainedTokenizer

class TimeSeriesTokenizeFunction(CachableTokenizeFunction):
    """时序数据的 TokenizeFunction

    Args:
        tokenizer (PreTrainedTokenizer): 分词器
        media_root (str | Path): 多模态文件的根目录
        max_length (int | None): 最大序列长度，超长部分会被截断
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        media_root: str | Path,
        max_length: int | None = None,
    ):
        self.max_length = max_length
        self.media_root = media_root

        # 获取特殊 token 的 ID
        self.ts_start_token_id = tokenizer.convert_tokens_to_ids("<|ts|>")
        self.ts_context_token_id = tokenizer.convert_tokens_to_ids("<TS_CONTEXT>")
        self.ts_end_token_id = tokenizer.convert_tokens_to_ids("<|/ts|>")

        super().__init__(tokenizer)

    def __call__(self, item: dict, **kwargs):
        """处理 JsonlDataset 解析出来的每一行数据"""
        if self._is_time_series(item["messages"]):
            return self.time_series_get_item(item)
        else:
            return self.pure_text_get_item(item)
```

**关键方法说明：**

1. **`_is_time_series()`**: 判断数据是否包含时序信号
2. **`collect_time_series_paths_and_extra()`**: 从 messages 中提取时序文件路径和元信息
3. **`build_ts_transform()`**: 构建时序数据的加载和预处理函数，支持多种格式（.wav, .mp3, .csv, .npy）
4. **`replace_ts_token_naive()`**: 将文本中的 `<TS_CONTEXT>` 占位符替换为实际的 token 序列（`<|ts|><TS_CONTEXT>...<TS_CONTEXT><|/ts|>`）
5. **`time_series_get_item()`**: 处理多模态数据，返回包含时序特征的字典
6. **`pure_text_get_item()`**: 处理纯文本数据

**返回格式：**

`tokenize_fn` 必须返回包含 `input_ids` 和 `labels` 两个字段的字典，额外的多模态数据字段会在 `collate_fn` 中被收集。

```python
{
    "input_ids": [...],              # token IDs（必需）
    "labels": [...],                 # 训练标签（必需）
    "time_series_signals": tensor,   # 时序信号（多模态数据专属）
    "ts_len": int,                   # 信号长度（多模态数据专属）
    "ts_sr": int,                    # 采样率（多模态数据专属）
    "num_tokens": int,               # token 总数
    "ts_bos_token": int,             # 特殊 token IDs
    "ts_eos_token": int,
    "ts_context_token": int,
}
```

**对话模板处理：**

用户需要自行实现对话模板的应用逻辑。本示例中使用 `naive_apply_chat_template()` 方法，将多轮对话转换为模型输入格式：

```python
def naive_apply_chat_template(self, messages: list[dict], tokenizer):
    """应用对话模板

    将对话数据转换为模型需要的格式：
    <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>
    """
    # 实现对话模板逻辑
    # 返回 input_ids 和 labels
```

### CollateFunction 实现

`collate_fn` 对应 PyTorch DataLoader 的 collate 函数，负责将一个 batch 的数据整合在一起。主要完成两件事：

1. **文本数据的 batch 聚合**：将多条样本的文本数据打包成 batch
2. **多模态数据的聚合**：收集 batch 中所有的多模态数据

**实现示例：**

```python
from xtuner.v1.datasets.collator import build_text_ctx_labels

def ts_sft_collator(
    instances: list[list[dict]],
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
) -> list[dict]:
    """时序数据的 collate 函数

    Args:
        instances (list[list[dict]]): batch 数据，每个元素是一条 packed 数据，由多个 tokenize_fn 的输出组成
        pack_max_length (int): 打包的最大长度
        padding_token_idx (int): padding token 的 ID
        pack_to_max_length (bool): 是否打包到最大长度

    Returns:
        list[dict]: 整合后的 batch 数据，每个元素包含 seq_ctx 和 shifted_labels
    """
    ret: list[dict] = []
    instance0 = instances[0][0]
    ts_bos_token = instance0["ts_bos_token"]
    ts_eos_token = instance0["ts_eos_token"]
    ts_context_token = instance0["ts_context_token"]

    for instance in instances:
        # 构建文本的 SequenceContext 和 labels
        text_seq_ctx, shifted_labels = build_text_ctx_labels(
            instance=instance,
            pack_max_length=pack_max_length,
            padding_token_idx=padding_token_idx,
            pack_to_max_length=pack_to_max_length,
        )

        # 收集时序数据
        ts_values = [
            i["time_series_signals"] for i in instance if "time_series_signals" in i
        ]
        ts_lens = [i["ts_len"] for i in instance if "ts_len" in i]
        ts_sr = [i["ts_sr"] for i in instance if "ts_sr" in i]

        # 处理时序数据
        if ts_values:
            ts_lens = torch.tensor(ts_lens)
            sr = torch.tensor(ts_sr)
            time_series_signals = ts_values
        else:
            time_series_signals = None
            ts_lens = None
            sr = None

        # 构建扩展的 SequenceContext
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

**核心流程：**

1. 使用 XTuner 提供的 `build_text_ctx_labels()` 处理文本部分的 batch 聚合（前提是 `tokenize_fn` 返回的对象包含 `input_ids` 和 `labels` 字段）
2. 从 batch 中收集所有样本的多模态数据（如时序信号、长度、采样率）
3. 构建扩展的 `TimeSeriesSequenceContext`，将文本和多模态数据整合在一起
4. 返回包含 `seq_ctx` 和 `shifted_labels` 的字典

### SequenceContext 扩展

`SequenceContext` 是 XTuner 模型接受的数据类型。在 `collate_fn` 中需要将文本 `SequenceContext` 与多模态数据整合成扩展的 `SequenceContext`。

**实现示例：**

```python
from xtuner.v1.data_proto import SequenceContext
import torch

class TimeSeriesSequenceContext(SequenceContext):
    """时序数据的 SequenceContext 扩展

    Args:
        time_series_signals (list[torch.FloatTensor] | torch.FloatTensor | None): 时序信号数据
        ts_lens (torch.Tensor | None): 时序信号的长度
        ts_sr (torch.Tensor | None): 时序信号的采样率
        ts_bos_token (int): 时序数据的开始 token ID
        ts_eos_token (int): 时序数据的结束 token ID
        ts_context_token (int): 时序特征占位 token ID
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

**设计要点：**

- 继承 `SequenceContext` 基类
- 添加模态特定的字段（如时序信号、长度、采样率等）
- 添加特殊 token 的 ID（用于标记模态数据的边界和占位符）
- 通过 `**data` 传递基类所需的文本相关字段

______________________________________________________________________

## 模型组件

模型组件需要实现两个类：`ModelConfig` 和 `Model`。

### 模型配置

继承 `XTunerBaseModelConfig`，定义模型的配置参数。

**实现示例：**

```python
from xtuner.v1.model.compose.base import XTunerBaseModelConfig
from xtuner.v1.model.base import TransformerConfig

class TsModelConfig(XTunerBaseModelConfig):
    """时序模型配置

    Args:
        text_config (TransformerConfig): 文本模型的配置
        ts_model_path (str | Path): 时序 encoder 的路径
    """
    text_config: TransformerConfig
    ts_model_path: str | Path

    def build(self):
        """构建模型实例"""
        return TSModel(self)

    @property
    def hf_config(self):
        """Huggingface 配置转换

        对于使用 trust_remote_code 的模型：
            - 返回 None 即可，XTuner 会自动拷贝原始目录下的配置文件

        对于 Transformers 仓库内置的模型：
            - 需要实现 XTuner Config 到 Transformers Config 的转换
        """
        return None
```

**关键点：**

1. **`build()` 方法**：用于构建模型实例
2. **`hf_config` 属性**：
   - 对于使用 `trust_remote_code` 的模型（如本例），返回 `None` 即可，XTuner 会自动拷贝原始目录下的配置文件到保存目录
   - 对于 Transformers 仓库内置的模型，需要实现 XTuner Config 到 Transformers Config 的转换

### 模型实现

继承 `XTunerBaseModel`，实现模型的前向传播和保存加载逻辑。

**实现示例：**

```python
from xtuner.v1.model.base import BaseModel as XTunerBaseModel
from transformers import AutoModel, AutoConfig

class TSModel(XTunerBaseModel):
    """时序多模态模型

    Args:
        config (TsModelConfig): 模型配置
    """
    def __init__(self, config: TsModelConfig):
        super().__init__(config)

        # 加载语言模型
        self.language_model = config.text_config.build()
        self.language_model.requires_grad_(False)  # 冻结 LLM

        # 【关键】必须在 cuda device 上下文中初始化模型
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

**关键点：**

1. **必须在 `__init__` 中使用 `cuda` device 上下文初始化模型**（非常重要）
2. 需要实现 `forward()` 方法
3. 需要实现 `from_hf()` 和 `save_hf()` 方法用于模型加载和保存

### 关键接口

#### 1. `forward()` 方法

实现模型的前向传播逻辑。参考代码模板中的实现。

**核心逻辑：**

1. 使用 `embed_tokens()` 获取文本 token 的 embedding
2. 使用多模态 encoder 提取特征
3. 将多模态特征替换到 embedding 中占位符 token 的位置
4. 将融合后的 embedding 传递给语言模型

**实现示例：**

```python
def forward(
    self, seq_ctx: TimeSeriesSequenceContext, loss_ctx: CELossContext
) -> MoEModelOutputs:
    """前向传播

    Args:
        seq_ctx (TimeSeriesSequenceContext): 输入的序列上下文
        loss_ctx (CELossContext): 损失函数上下文

    Returns:
        MoEModelOutputs: 模型输出
    """
    input_ids = seq_ctx.input_ids

    # 1. 获取文本的 embedding
    inputs_embeds = self.language_model.embed_tokens(input_ids)

    # 2. 处理时序数据
    time_series_signals = seq_ctx.time_series_signals
    if time_series_signals is not None:
        # 2.1 使用时序 encoder 提取特征
        ts_features, ts_pad_mask = self.get_ts_feature(
            time_series_signals, seq_ctx.ts_lens, seq_ctx.ts_sr
        )
        ts_features = ts_features[~ts_pad_mask].to(
            inputs_embeds.device, inputs_embeds.dtype
        )

        # 2.2 将时序特征替换到 embedding 中的占位符位置
        B, N, C = inputs_embeds.shape
        input_ids = input_ids.reshape(B * N)
        inputs_embeds = inputs_embeds.reshape(B * N, C)

        ts_placeholder = input_ids == seq_ctx.ts_context_token
        inputs_embeds[ts_placeholder] = ts_features
        inputs_embeds = inputs_embeds.reshape(B, N, C)
    else:
        # 纯文本数据：使用 fake 时序数据保持梯度图一致
        fake_time_series_signals = torch.zeros(
            (1, 147, 3), device=input_ids.device, dtype=inputs_embeds.dtype
        )
        fake_ts_lens = torch.tensor([147], device=input_ids.device)
        fake_sr = torch.tensor([36], device=input_ids.device)
        ts_features, _ = self.get_ts_feature(
            fake_time_series_signals, fake_ts_lens, fake_sr
        )
        inputs_embeds = inputs_embeds * 0.0 + ts_features.sum() * 0.0

    # 3. 构建语言模型的输入
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

    # 4. 调用语言模型
    outputs = self.language_model(lang_seq_ctx, loss_ctx)
    return outputs
```

#### 2. `from_hf()` 方法

从 Huggingface checkpoint 加载模型权重。参考代码模板中的实现。

**实现示例：**

```python
def from_hf(
    self, hf_path: str | Path, strict: bool = True
) -> tuple[set[str], set[str], set[str]]:
    """从 HF checkpoint 加载权重

    Args:
        hf_path (str | Path): checkpoint 路径
        strict (bool): 是否严格匹配权重

    Returns:
        tuple: (missing_keys, unexpected_keys, error_keys)
    """
    self.language_model.from_hf(hf_path, strict)
    return set(), set(), set()
```

#### 3. `save_hf()` 方法

保存模型为 Huggingface 格式。参考代码模板中的实现。

**实现示例：**

```python
def save_hf(
    self,
    hf_dir: Path | str,
    save_dtype: torch.dtype = torch.bfloat16,
    safetensors_prefix: str = "model",
):
    """保存模型到 HF 格式

    Args:
        hf_dir (Path | str): 保存目录
        save_dtype (torch.dtype): 保存的数据类型
        safetensors_prefix (str): safetensors 文件前缀
    """
    self.unshard()
    hf_dir = Path(hf_dir)

    # 1. 保存语言模型
    self.language_model.save_hf(hf_dir, save_dtype, "HF")

    # 2. 保存时序 encoder
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

        # 3. 更新 weight_map
        with open(hf_dir / "model.safetensors.index.json", "w") as f:
            json.dump(
                {"weight_map": weight_map_dict, "metadata": {}}, f, indent=2
            )

    dist.barrier()
    return (set(self.state_dict()), set(), set())
```

**保存逻辑：**

1. 保存语言模型权重
2. 收集并重映射时序 encoder 权重（添加 `model.time_series.` 前缀）
3. 更新 `model.safetensors.index.json` 中的 weight_map

______________________________________________________________________

## 训练配置

基于实现的数据和模型组件，配置训练参数。

**配置示例（xtuner_project/config.py）：**

```python
from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner_project.model import TsModelConfig
from xtuner_project.data import TimeSeriesTokenizeFnConfig

# 路径配置
model_path = "/path/to/pretrained/model"
work_dir = "/path/to/work_dir"
meta_data_path = "/path/to/data.jsonl"

# 训练超参数
sample_max_length = 8192
pack_max_length = 8192
global_batch_size = 8
total_epoch = 1
lr = 2e-5
lr_min = 2e-6
weight_decay = 0.05
warmup_ratio = 0.03

# 模型配置
model_cfg = TsModelConfig(
    ts_model_path=model_path,
    text_config=Qwen3MoE30BA3Config(
        max_position_embeddings=32768,
        n_routed_experts=512,
        hf_key_mapping={r"^model.": "model.language_model."},
    ),
)

# 数据配置
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

# 优化器和学习率配置
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)

# FSDP 配置
fsdp_cfg = FSDPConfig(
    recompute_ratio=1.0,
    ep_size=1,
    torch_compile=False,
)

# Trainer 配置
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

**配置要点：**

1. **模型配置**：

   - 指定时序 encoder 路径（`ts_model_path`）
   - 配置语言模型（`text_config`）
   - 设置权重映射规则（`hf_key_mapping`）

2. **数据配置**：

   - 使用 `VLMJsonlDataset` 读取 jsonl 数据
   - 指定自定义的 `tokenize_fn` 配置类
   - 指定自定义的 `collator` 函数路径

3. **训练配置**：

   - 优化器：AdamW
   - 学习率调度：cosine with warmup
   - FSDP：并行配置
   - 损失函数：交叉熵（chunk mode, 节省显存）

______________________________________________________________________

## 启动训练

使用 XTuner 提供的训练脚本启动训练：

```bash
# 单卡训练
torchrun --master-port 29501 --nproc-per-node 8 -m  xtuner.v1.train.cli.sft --config xtuner_project/config.py
```

______________________________________________________________________

## 总结

本仓库展示了如何将 Huggingface Encoder 与 XTuner LLM 结合进行多模态训练：

1. **数据层面**：扩展 `SequenceContext`，实现自定义的 `TokenizeFunction` 和 `CollateFunction`
2. **模型层面**：实现 `ModelConfig` 和 `Model`，重点是 `forward()`、`save_hf()`、`from_hf()` 方法
3. **配置层面**：整合数据和模型配置，设置训练超参数

用户可以参考本实践，适配其他类型的 Huggingface Encoder（如视觉 encoder、音频 encoder 等），快速搭建自己的多模态训练流程。

______________________________________________________________________
