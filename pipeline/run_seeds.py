from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
RUN_PIPELINE = PIPELINE_DIR / "run_pipeline.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_tasks import PAPER_TASKS

def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

def available_seed_dirs(scaffold_root: Path, task: str) -> List[int]:
    task_dir = Path(scaffold_root) / task
    if not task_dir.is_dir():
        task_dir = Path(scaffold_root) / "scaffold" / task
    if not task_dir.is_dir():
        return []
    seeds = []
    for path in task_dir.glob("seed*"):
        if path.is_dir() and (path / "train.csv").is_file():
            try:
                seeds.append(int(path.name.replace("seed", "")))
            except ValueError:
                continue
    return sorted(seeds)

def parse_seed_spec(spec: str, scaffold_root: Path, task: str) -> List[int]:

    text = str(spec).strip().lower()
    if text in {"random10", "ten", "baseline10", "scaffold10", "split10"}:
        seeds = available_seed_dirs(scaffold_root, task)
        if len(seeds) >= 10:
            return seeds[:10]
        log(
            f"fewer than ten per-seed splits found for {task}; "
            "falling back to the confirmed ten seeds 42..132"
        )
        return [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
    if "-" in text and "," not in text:
        left, right = text.split("-", 1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in text.split(",") if part.strip()]

def run_seed(seed: int, seed_dir: Path, args: argparse.Namespace, config: Dict[str, Any]) -> int:
    command = [
        sys.executable, str(RUN_PIPELINE),
        "--stages", args.stages_inner,
        "--run-dir", str(seed_dir),
        "--seed", str(seed),
        "--task", args.task,
        "--smoke-task", args.smoke_task,
        "--scaffold-root", args.scaffold_root,
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--pooling", args.pooling,
        "--train-variant", args.train_variant,
        "--full-data",
    ]
    if args.pretrained_checkpoint:
        command += ["--pretrained-checkpoint", args.pretrained_checkpoint]
    if args.enforced:
        command += ["--enforced"]
    if args.random_smiles:
        command += ["--random-smiles"]
    if args.full_eval:
        command += ["--full-eval"]
    log(f"seed {seed}: {' '.join(command[:6])} ... (log: {seed_dir / 'logs'})")
    started = time.time()
    completed = subprocess.run([str(part) for part in command], cwd=str(REPO_ROOT))
    log(f"seed {seed}: exit={completed.returncode} ({time.time() - started:.0f}s)")
    return completed.returncode

def collect(seed_dirs: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed_dir in seed_dirs:
        path = seed_dir / "05_eval.json"
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for item in json.load(handle):
                details = item.get("details", {}) or {}
                for metric in ("AUC",):
                    value = details.get(metric)
                    if value is None:
                        continue
                    rows.append({
                        "seed": int(item.get("seed", 0)),
                        "source": item.get("source", ""),
                        "variant": item.get("variant", ""),
                        "split": item.get("split", ""),
                        "task": item.get("task", ""),
                        "metric": metric,
                        "value": float(value),
                        "checkpoint": item.get("checkpoint", ""),
                    })
    return rows

def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[float]] = {}
    for row in rows:
        key = (row["source"], row["variant"], row["split"], row["task"], row["metric"])
        buckets.setdefault(key, []).append(row["value"])
    summary = []
    for (source, variant, split, task, metric), values in sorted(buckets.items()):
        summary.append({
            "source": source,
            "variant": variant,
            "split": split,
            "task": task,
            "metric": metric,
            "n_seeds": len(values),
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        })
    return summary

def write_outputs(root: Path, rows: List[Dict[str, Any]], summary: List[Dict[str, Any]],
                  args: argparse.Namespace, seeds: List[int], failures: List[int]) -> None:
    root.mkdir(parents=True, exist_ok=True)

    with open(root / "seed_summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "variant", "split", "task", "metric", "n_seeds",
                        "mean", "sd", "min", "max"],
        )
        writer.writeheader()
        writer.writerows(summary)

    with open(root / "seed_metrics_long.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "source", "variant", "split", "task", "metric",
                        "value", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)

    per_seed = []
    for row in rows:
        if row["metric"] != "AUC":
            continue
        per_seed.append({
            "task": row["task"],
            "seed": row["seed"],
            "source": row["source"],
            "variant": row["variant"],
            "split": row["split"],
            "score": row["value"],
            "metric": row["metric"],
            "checkpoint": row["checkpoint"],
        })
    with open(root / f"{args.task}_seed_results.json", "w", encoding="utf-8") as handle:
        json.dump(per_seed, handle, indent=2)

    lines = ["# Multi-seed run summary", ""]
    lines.append(f"- task: `{args.task}`")
    lines.append(f"- scaffold root: `{args.scaffold_root}`")
    lines.append(f"- seeds: {seeds}")
    lines.append(f"- variant: `{args.train_variant}`")
    lines.append(f"- epochs per seed: {args.epochs}, seq_len: {args.seq_len}")
    lines.append(f"- failures: {failures if failures else 'none'}")
    lines.append("")
    lines.append("| source | variant | split | metric | n seeds | mean | SD | min | max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary:
        lines.append(
            f"| {row['source']} | {row['variant']} | {row['split']} | {row['metric']} | "
            f"{row['n_seeds']} | {row['mean']:.4f} | {row['sd']:.4f} | "
            f"{row['min']:.4f} | {row['max']:.4f} |"
        )
    lines.append("")
    (root / "seed_report.md").write_text("\n".join(lines), encoding="utf-8")

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STGFormer pipeline over several seeds.")
    parser.add_argument("--task", choices=PAPER_TASKS, default="bace")
    parser.add_argument("--smoke-task", default="")
    parser.add_argument("--scaffold-root", default="fintune_datas/moleculenet_random_scaffold")
    parser.add_argument(
        "--seeds",
        default="42,52,62,72,82,92,102,112,122,132",
        help="ten paper runs: 42,52,62,72,82,92,102,112,122,132; or random10",
    )
    parser.add_argument("--train-variant", default="STG+PE")
    parser.add_argument("--pretrained-checkpoint", default="checkpoints/pretraining/Bert_STG+PE_1000k.pt")
    parser.add_argument("--enforced", action="store_true",
                        help="use enforced CSV data for all available splits")
    parser.add_argument("--random-smiles", action="store_true",
                        help="use enforced/random-SMILES training data only")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--stages-inner", default="data,train,eval")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--full-eval", action="store_true")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    scaffold_root = Path(args.scaffold_root)
    if not scaffold_root.is_absolute():
        scaffold_root = REPO_ROOT / scaffold_root
    if not args.smoke_task:
        args.smoke_task = f"{args.task}_smoke"

    seeds = parse_seed_spec(args.seeds, scaffold_root, args.task)
    root = Path(args.run_dir) if args.run_dir else (
        REPO_ROOT / "runs" / f"seeds_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STGFormer multi-seed run")
    print("=" * 78)
    log(f"task={args.task} seeds={seeds}")
    log(f"scaffold root: {scaffold_root}")
    log(f"run root: {root}")

    seed_dirs: List[Path] = []
    failures: List[int] = []
    for seed in seeds:
        seed_dir = root / f"seed{seed}"
        seed_dirs.append(seed_dir)
        code = run_seed(seed, seed_dir, args, {})
        if code != 0:
            failures.append(seed)

    rows = collect(seed_dirs)
    summary = aggregate(rows)
    write_outputs(root, rows, summary, args, seeds, failures)

    print()
    print("=" * 78)
    print("MULTI-SEED DONE")
    print("=" * 78)
    print(f"seeds run        : {len(seeds)} ({len(failures)} failed)")
    print(f"per-seed metrics : {root / 'seed_summary.csv'}")
    print(f"report           : {root / 'seed_report.md'}")
    print(f"per-seed results : {root / f'{args.task}_seed_results.json'}")
    for row in summary:
        print(f"  {row['source']:20s} {row['variant']:7s} {row['split']:5s} {row['metric']:4s} "
              f"n={row['n_seeds']:<3d} mean={row['mean']:.4f} sd={row['sd']:.4f}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
