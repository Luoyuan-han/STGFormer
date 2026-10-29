from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_tasks import PAPER_TASKS, is_smoke_task
from data_io import find_task_dir, has_aggregate_csv, has_split_file, read_split

VARIANT_SPECS: Dict[str, Dict[str, bool]] = {
    "Base": {"stg": False, "learnable_pos": False},
    "PE": {"stg": False, "learnable_pos": True},
    "STG": {"stg": True, "learnable_pos": False},
    "STG+PE": {"stg": True, "learnable_pos": True},
}

def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def task_root(scaffold_root: Path, task: str) -> Path:
    return find_task_dir(scaffold_root, task)

def resolve_split_dir(scaffold_root: Path, task: str, seed: int) -> Path:

    task_dir = task_root(scaffold_root, task)
    seed_dir = task_dir / f"seed{seed}"
    if has_split_file(seed_dir, "train") or has_aggregate_csv(seed_dir):
        return seed_dir
    if has_split_file(task_dir, "train") or has_aggregate_csv(task_dir):
        return task_dir
    candidates = sorted(
        p for p in task_dir.glob("seed*")
        if has_split_file(p, "train") or has_aggregate_csv(p)
    )
    if candidates:
        print(f"WARNING: seed{seed} not found for {task}, using {candidates[0].name} instead.")
        return candidates[0]
    raise FileNotFoundError(f"no split files for task={task} under {scaffold_root}")

def load_split(split_dir: Path, name: str, enforced: bool = False,
               random_smiles: bool = False) -> List[List[Any]]:
    use_random_smiles = random_smiles and name == "train"
    rows, path = read_split(Path(split_dir), name, enforced=enforced,
                            random_smiles=use_random_smiles)
    print(f"[train_variant] loaded {len(rows)} {name} rows from {path}", flush=True)
    return rows

def build_config(variant: str, args: argparse.Namespace, rows: List[List[Any]]) -> Any:

    from config import BertConfig, configure_finetune

    if variant not in VARIANT_SPECS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANT_SPECS)}")

    config = BertConfig()
    spec = VARIANT_SPECS[variant]
    config.stg = spec["stg"]
    config.learnable_pos = spec["learnable_pos"]
    config.rope = False

    label_dim = max(len(row[1]) for row in rows)
    configure_finetune(
        config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        pooling=args.pooling,
        early_stop_metric=args.early_stop_metric,
        use_lr_scheduler=not args.no_scheduler,
        label_dim=label_dim,
    )
    return config

def load_state_dict(path: str):
    import torch

    if not path:
        print("No pretrained checkpoint supplied; training from scratch.", flush=True)
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    print(f"Loading pretrained checkpoint: {path}", flush=True)
    return torch.load(path, map_location="cpu", weights_only=True)

def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from Bert_trainer import STGFormerTrainer
    from Dataset import BertDataset

    if args.task not in PAPER_TASKS and not is_smoke_task(args.task):
        raise ValueError(
            f"unsupported task {args.task!r}; expected one of {', '.join(PAPER_TASKS)}"
        )
    set_seed(args.seed)
    out_dir = Path(args.out_dir).resolve()
    scaffold_root = Path(args.scaffold_root).resolve()
    split_dir = resolve_split_dir(scaffold_root, args.task, args.seed)

    train_rows = load_split(split_dir, "train", args.enforced, args.random_smiles)
    valid_rows = load_split(split_dir, "valid", args.enforced)
    test_rows = load_split(split_dir, "test", args.enforced)

    config = build_config(args.variant, args, train_rows)
    config.checkpoint_dir = str(out_dir / "checkpoints")
    config.log_dir = str(out_dir / "logs")
    print(
        f"[train_variant] variant={args.variant} task={args.task} "
        f"STG={config.stg} learnable_pos={config.learnable_pos} "
        f"seq_len={config.seq_len} reg={config.reg} tgt_size={config.tgt_size} "
        f"train={len(train_rows)} valid={0 if valid_rows is None else len(valid_rows)} "
        f"test={len(test_rows)}",
        flush=True,
    )
    print(f"[train_variant] split dir: {split_dir}", flush=True)

    started = time.time()
    if args.mode == "train":
        state_dict = load_state_dict(args.pretrained)
        tr = BertDataset(train_rows, config)
        va = BertDataset(valid_rows, config) if valid_rows is not None else None
        te = BertDataset(test_rows, config)
        trainer = STGFormerTrainer(config, state_dict=state_dict, seed=args.seed, ind=0)
        score = trainer.train_and_test(tr, te, va, cls_w=None)
        details = dict(getattr(trainer, "last_eval_details", {}) or {})
        if not details:

            details = {
                "split": "valid",
                "AUC": float(score),
            }
        checkpoint_path = Path(trainer.early_stopping.path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode eval")
        state_dict = load_state_dict(args.checkpoint)
        trainer = STGFormerTrainer(config, state_dict=state_dict, seed=args.seed, ind=0)
        dataset = BertDataset(test_rows if args.split == "test" else valid_rows, config)
        if config.reg and config.z:
            norm_rows = BertDataset(train_rows, config)
            mean, sigma = norm_rows.z_score()
            dataset.set_z_score(mean, sigma)
        score, details = trainer.evaluate_downstream(dataset, split_name=args.split)
        checkpoint_path = Path(args.checkpoint)

    result = {
        "mode": args.mode,
        "variant": args.variant,
        "task": args.task,
        "split": args.split if args.mode == "eval" else details.get("split", "valid"),
        "seed": args.seed,
        "stg": bool(config.stg),
        "learnable_pos": bool(config.learnable_pos),
        "seq_len": int(config.seq_len),
        "batch_size": int(config.batch_size),
        "pooling": config.pooling,
        "split_dir": str(split_dir),
        "n_train": len(train_rows),
        "n_valid": 0 if valid_rows is None else len(valid_rows),
        "n_test": len(test_rows),
        "score": float(score),
        "details": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in details.items()},
        "checkpoint": str(checkpoint_path),
        "pretrained": args.pretrained if args.mode == "train" else args.checkpoint,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "elapsed_sec": round(time.time() - started, 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.mode}_{args.variant.replace('+', 'p')}.json"
    with open(out_dir / name, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train / evaluate one STGFormer variant")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--variant", choices=sorted(VARIANT_SPECS), default="STG+PE")
    parser.add_argument("--scaffold_root", required=True)
    parser.add_argument("--task", default="bace")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--early_stop_metric", choices=["loss", "auc"], default="loss")
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument("--enforced", action="store_true",
                        help="use enforced CSV data for all available splits")
    parser.add_argument("--random-smiles", action="store_true",
                        help="use the enforced/random-SMILES training CSV when available")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)

if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    run(parse_args())
