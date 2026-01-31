import os
import shutil

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig

from xtuner_project.data import TimeSeriesTokenizeFnConfig
from xtuner_project.model import TsModelConfig


# 路径配置
meta_data_path = "/path/to/data-meta"
model_path = "/path/to/work_dir"
work_dir = "/path/to/work_dir"


# 将当前配置文件拷贝到work_dir
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
current_file = __file__
shutil.copy(current_file, work_dir)

# 训练超参数
sample_max_length = 8192
pack_max_length = 8192
processor_path = model_path
num_workers = 0
global_batch_size = 8
total_epoch = 1
hf_interval = 10
hf_max_keep = 5
checkpoint_interval = 500
checkpoint_maxkeep = 5
lr = 2e-5
lr_min = 2e-6
weight_decay = 0.05
warmup_ratio = 0.03
recompute_ratio = 1.0
loss_reduction = "square"
enable_3d_rope = False

# model config
model_cfg = TsModelConfig(
    ts_model_path=model_path,
    text_config=Qwen3MoE30BA3Config(
        max_position_embeddings=32768,
        n_routed_experts=512,
        hf_key_mapping={r"^model.": "model.language_model."},
    ),
)

model_cfg.compile_cfg = False

model_cfg.text_config.rope_scaling_cfg = RopeScalingConfig(
    fope_init_factor=0.1,
    fope_sep_head=True,
    num_inv_freq=None,
)
model_cfg.text_config.vocab_size = 155008

model_cfg.text_config.router.use_grouped_router = True
model_cfg.text_config.router.router_n_groups = 4

# ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
# for name, _data in ds_collections.items():
_data_cfg = {
    "dataset": DatasetConfig(
        name="11",
        anno_path="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/train_1000_samples.jsonl",
        media_root="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/",
        sample_ratio=100,
        enable_sequential_sampler=True,
        class_name="VLMJsonlDataset",
    ),
    "tokenize_fn": TimeSeriesTokenizeFnConfig(
        media_root="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/",
        max_length=sample_max_length,
    ),
}
dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    collator="xtuner_project.data.ts_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(
    recompute_ratio=recompute_ratio,
    ep_size=1,
    torch_compile=False,
    checkpoint_preserve_rng_state=False,
)

resume_cfg = ResumeConfig(auto_resume=False)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=10,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
    strict_load=False,
)
