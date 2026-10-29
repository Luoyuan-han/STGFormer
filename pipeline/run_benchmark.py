from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
TRAIN_VARIANT = PIPELINE_DIR / "train_variant.py"
DEFAULT_SEEDS = "42,52,62,72,82,92,102,112,122,132"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_tasks import PAPER_TASKS

def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]

def run_command(command: List[str], cwd: Path, log_path: Path) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("exec:", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,

            stderr=None,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def command_base(args: argparse.Namespace, seed: int, out_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(TRAIN_VARIANT),
        "--variant", args.variant,
        "--scaffold_root", str(Path(args.scaffold_root).resolve()),
        "--task", args.task,
        "--out_dir", str(out_dir.resolve()),
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--pooling", args.pooling,
        "--seed", str(seed),
    ]

def run_one(args: argparse.Namespace, root: Path, early_metric: str, seed: int) -> Dict[str, Any]:
    run_dir = root / early_metric / f"seed{seed}"
    train_dir = run_dir / "train"
    eval_dir = run_dir / "eval"
    result_name = args.variant.replace("+", "p")
    eval_result = eval_dir / f"eval_{result_name}.json"

    if args.resume and eval_result.is_file():
        print(f"resume: {early_metric} seed={seed}", flush=True)
        result = read_json(eval_result)
    else:
        train_command = command_base(args, seed, train_dir) + [
            "--mode", "train",
            "--early_stop_metric", early_metric,
        ]
        if args.pretrained_checkpoint:
            train_command += ["--pretrained", str(Path(args.pretrained_checkpoint).resolve())]
        if args.no_scheduler:
            train_command.append("--no_scheduler")
        run_command(train_command, train_dir, run_dir / "train.log")

        train_result = read_json(train_dir / f"train_{result_name}.json")
        checkpoint = Path(train_result["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"best checkpoint not found: {checkpoint}")

        eval_command = command_base(args, seed, eval_dir) + [
            "--mode", "eval",
            "--split", "test",
            "--checkpoint", str(checkpoint),
            "--early_stop_metric", early_metric,
        ]
        if args.no_scheduler:
            eval_command.append("--no_scheduler")
        run_command(eval_command, eval_dir, run_dir / "eval.log")
        result = read_json(eval_result)

    result["early_stop_metric"] = early_metric
    result["seed"] = seed
    return result

def write_summary(root: Path, args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for result in results:
        details = result.get("details", {}) or {}
        metric_name = "AUC"
        metric_value = details.get(metric_name)
        if metric_value is None:
            raise ValueError(f"missing test AUC in result: {result}")
        rows.append({
            "task": args.task,
            "variant": args.variant,
            "early_stop_metric": result["early_stop_metric"],
            "seed": result["seed"],
            "test_metric": metric_name,
            "test_value": float(metric_value),
            "checkpoint": result.get("checkpoint", ""),
        })

    with open(root / "benchmark_results.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(root / "benchmark_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    lines = [f"# STGFormer benchmark: {args.task}", ""]
    lines += [
        f"- variant: `{args.variant}`",
        f"- seeds: `{args.seeds}`",
        f"- epochs: `{args.epochs}`",
        f"- batch size: `{args.batch_size}`",
        f"- learning rate: `{args.lr}`",
        "",
        "| early stopping | test metric | n | mean | SD |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for early_metric in parse_csv(args.early_stop_metrics):
        subset = [row for row in rows if row["early_stop_metric"] == early_metric]
        if not subset:
            continue
        values = [row["test_value"] for row in subset]
        lines.append(
            f"| {early_metric} | {subset[0]['test_metric']} | {len(values)} | "
            f"{statistics.fmean(values):.6f} | "
            f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f} |"
        )
    (root / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one STGFormer task with loss/AUC early stopping."
    )
    parser.add_argument("--task", choices=PAPER_TASKS, required=True)
    parser.add_argument("--scaffold-root", default="fintune_datas/moleculenet_random_scaffold")
    parser.add_argument("--pretrained-checkpoint", default=os.environ.get("STGFORMER_PRETRAINED", "checkpoints/pretraining/Bert_STG+PE_1000k.pt"))
    parser.add_argument("--variant", choices=["Base", "PE", "STG", "STG+PE"], default="STG+PE")
    parser.add_argument("--early-stop-metrics", default="loss,auc")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    scaffold_root = Path(args.scaffold_root).expanduser()
    args.scaffold_root = str(scaffold_root if scaffold_root.is_absolute() else (REPO_ROOT / scaffold_root).resolve())
    pretrained = Path(args.pretrained_checkpoint).expanduser() if args.pretrained_checkpoint else None
    if pretrained is not None and not pretrained.is_absolute():
        args.pretrained_checkpoint = str((REPO_ROOT / pretrained).resolve())
    seeds = [int(value) for value in parse_csv(args.seeds)]
    early_metrics = parse_csv(args.early_stop_metrics)
    invalid = [value for value in early_metrics if value not in {"loss", "auc"}]
    if invalid:
        raise ValueError(f"invalid early-stop metrics: {invalid}")
    if args.pretrained_checkpoint and not Path(args.pretrained_checkpoint).is_file():
        raise FileNotFoundError(args.pretrained_checkpoint)

    root = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "runs" /
        f"benchmark_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"task={args.task} metrics={early_metrics} seeds={seeds}")
    print(f"output={root}")
    if args.dry_run:
        return 0

    root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    for early_metric in early_metrics:
        for seed in seeds:
            try:
                print(f"\n[{early_metric}] seed={seed}", flush=True)
                results.append(run_one(args, root, early_metric, seed))
            except Exception as exc:
                failures.append(f"{early_metric}/seed{seed}: {exc}")
                print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)

    if results:
        write_summary(root, args, results)
        print(f"report: {root / 'benchmark_report.md'}")
    if failures:
        (root / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
