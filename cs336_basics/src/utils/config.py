from dataclasses import dataclass, field, asdict
from typing import Optional, Iterable



@dataclass 
class ModelConfig:
    vocab_size: Optional[int] = field(default=50257)
    context_length: Optional[int] = field(default=1024)
    num_layers: Optional[int] = field(default=12)
    d_model: Optional[int] = field(default=768)
    num_heads: Optional[int] = field(default=12)
    d_ff: Optional[int] = field(default=3072)
    attn_pdrop: Optional[float] = field(default=0.1)
    residual_pdrop: Optional[float] = field(default=0.1)


@dataclass
class OptimizerConfig:
    lr_max: Optional[float] = field(default=5e-4)
    lr_min: Optional[float] = field(default=5e-6)
    warmup_iters: Optional[int] = field(default=0)
    total_iters: Optional[int] = field(default=2*(10**4))
    max_norm: Optional[float] = field(default=1.0)
    penalty: Optional[float] = field(default=0.001)
    loss_type: Optional[str] = field(default="exp")
    init_from: str = 'scratch'

@dataclass
class LoggingConfig:
    log_interval: Optional[int] = field(default=20)
    eval_interval: Optional[int] = field(default=200)
    eval_iters: Optional[int] = field(default=1200)
    wandb_logging: bool = True
    wandb_project: str = "Clip-GHM"
    wandb_path: str = "./others/wandb"
    raw: Optional[bool] = field(default=True)
    S3_upload: Optional[bool] = field(default=False)
    S3_bucket_name: Optional[str] = field(default='yuhangbucket')

@dataclass 
class UtilConfig(LoggingConfig, OptimizerConfig, ModelConfig):
    device: Optional[str] = field(default='cuda')
    batch_size: Optional[int] = field(default=128)


