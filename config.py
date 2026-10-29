from __future__ import annotations

from typing import Any, Optional

import torch

def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class BaseConfig:

    DEFAULTS = {
        "d_model": 768,
        "num_heads": 12,
        "dropout": 0.1,
        "decoder_layers": 12,
        "ffn": 4,
        "bias": True,
        "vocab_size": 156,
        "seq_len": 256,
        "tgt_size": 2,
        "stg": False,
        "MLM": False,
        "G_GPT": False,
        "reg": False,
        "mutcls": False,
        "learnable_pos": False,
        "rope": False,
        "scaffold_lead": False,
        "use_solvent": False,
        "z": False,
        "MAE": False,
        "accumulation_steps": 1,
        "train_steps": 50,
        "gen_trian_epoch": 50,
        "lr": 3e-5,
        "batch_size": 32,
        "use_lr_scheduler": True,
        "warmup_ratio": 0.0,
        "min_lr_ratio": 1.0,
        "early_stop_metric": "loss",
        "early_stopping_patience": 10,
        "pooling": "cls",
        "device": default_device(),
        "checkpoint_dir": "runs/default/checkpoints",
        "log_dir": "runs/default/logs",
        "loss_log_file": "bert_loss_curve.jsonl",
        "stream_num_workers": 4,
        "sample_batch_size": 64,
        "sample_size": 256,
        "sample_max_len": 128,
        "reply_buffer_sample": 100,
        "gpt_num_workers": 0,
        "gpt_to_2d": False,
    }

    def __init__(self, **overrides: Any) -> None:
        values = dict(self.DEFAULTS)
        values.update(overrides)
        self.__dict__.update(values)

    def update(self, **values: Any) -> "BaseConfig":
        self.__dict__.update(values)
        return self

    def copy(self) -> "BaseConfig":
        return type(self)(**self.__dict__)

class BertConfig(BaseConfig):

    def __init__(self, **overrides: Any) -> None:
        profile = {"dropout": 0.1, "seq_len": 256, "stg": False, "MLM": True,
                   "mutcls": True, "learnable_pos": True, "z": True,
                   "vocab_size": 156, "batch_size": 32, "lr": 3e-4,
                   "loss_log_file": "bert_loss_curve.jsonl"}
        super().__init__(**(profile | overrides))
        if (self.MLM or self.G_GPT) and "tgt_size" not in overrides:
            self.tgt_size = self.vocab_size

class GPTConfig(BaseConfig):

    def __init__(self, **overrides: Any) -> None:
        profile = {
            "dropout": 0.1, "seq_len": 128, "batch_size": 256, "lr": 3e-4,
            "stg": True, "G_GPT": True, "reg": True, "vocab_size": 123,
            "sample_max_len": 128, "sample_batch_size": 64, "sample_size": 256,
            "reply_buffer_sample": 100, "warmup_ratio": 0.05,
            "min_lr_ratio": 0.1, "early_stopping_patience": 5,
            "loss_log_file": "gpt_loss_curve.jsonl",
        }
        super().__init__(**(profile | overrides))
        self.tgt_size = self.vocab_size

def configure_finetune(config: BaseConfig, *, seq_len: int, batch_size: int,
                       epochs: int, lr: float, pooling: str,
                       early_stop_metric: str, use_lr_scheduler: bool,
                       label_dim: int, use_solvent: bool = False,
                       loss_log_file: Optional[str] = None) -> BaseConfig:

    config.update(
        MLM=False, G_GPT=False, use_solvent=bool(use_solvent),
        seq_len=int(seq_len), batch_size=int(batch_size),
        gen_trian_epoch=int(epochs), train_steps=int(epochs), lr=float(lr),
        early_stopping_patience=10, pooling=pooling,
        early_stop_metric=early_stop_metric,
        use_lr_scheduler=bool(use_lr_scheduler), reg=False, z=False,
        mutcls=int(label_dim) > 1,
        tgt_size=int(label_dim) if int(label_dim) > 1 else 1,
    )
    if loss_log_file:
        config.loss_log_file = loss_log_file
    return config

def configure_pretraining(config: BaseConfig, *, seq_len: int,
                          batch_size: int, epochs: int,
                          lr: float = 3e-4) -> BaseConfig:

    return config.update(
        MLM=True, G_GPT=False, reg=True, mutcls=False,
        early_stop_metric="loss", batch_size=int(batch_size),
        seq_len=int(seq_len), gen_trian_epoch=int(epochs),
        train_steps=int(epochs), lr=float(lr), use_lr_scheduler=True,
        warmup_ratio=0.0, min_lr_ratio=0.0,
        early_stopping_patience=5, tgt_size=config.vocab_size,
    )

def configure_generation(config: BaseConfig, *, seq_len: int,
                         batch_size: int, epochs: int, lr: float,
                         sample_batch_size: int, sample_max_len: int,
                         use_lr_scheduler: bool, gpt_num_workers: int = 0,
                         gpt_to_2d: bool = False) -> BaseConfig:

    return config.update(
        seq_len=int(seq_len), batch_size=int(batch_size),
        sample_batch_size=int(sample_batch_size),
        sample_max_len=int(sample_max_len), gen_trian_epoch=int(epochs),
        train_steps=int(epochs), lr=float(lr),
        use_lr_scheduler=bool(use_lr_scheduler),
        gpt_num_workers=int(gpt_num_workers), gpt_to_2d=bool(gpt_to_2d),
        tgt_size=config.vocab_size,
    )
