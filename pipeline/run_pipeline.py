from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_tasks import PAPER_TASKS
from data_io import find_task_dir, has_aggregate_csv, has_split_file, read_split

STAGE_ORDER = ["env", "sanity", "data", "train", "eval", "report"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "task": "bace",
    "smoke_task": "bace_smoke",
    "scaffold_root": "fintune_datas/moleculenet_random_scaffold",
    "seed": 42,
    "seq_len": 32,
    "batch_size": 32,
    "epochs": 1,
    "lr": 3e-5,
    "pooling": "cls",
    "train_variant": "STG+PE",
    "pretrained_checkpoint": "checkpoints/pretraining/Bert_STG+PE_1000k.pt",
    "enforced": False,
    "random_smiles": False,
    "full_data": False,
    "smoke_sizes": {"train": 4, "valid": 2, "test": 2},
    "full_eval_splits": [],
    "runs_dir": "runs",
}

class PipelineError(RuntimeError):
    pass

def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).is_file():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def run_command(command: List[Any], cwd: Path, log_file: Path,
                env: Optional[Dict[str, str]] = None) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log("exec: " + " ".join(str(part) for part in command))
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUNBUFFERED"] = "1"
    if env:
        child_env.update(env)
    with open(log_file, "w", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            stdout=handle,

            stderr=None,
            env=child_env,
        )
    if completed.returncode != 0:
        raise PipelineError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(str(part) for part in command)} (see {log_file})"
        )

def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(args.config) if args.config else PIPELINE_DIR / "pipeline_config.json"
    if config_path.is_file():
        file_config = read_json(config_path, {}) or {}
        config.update(file_config)
        log(f"loaded config file: {config_path}")
    if os.environ.get("STGFORMER_PRETRAINED"):
        config["pretrained_checkpoint"] = os.environ["STGFORMER_PRETRAINED"]

    for key in ("task", "smoke_task", "scaffold_root", "pretrained_checkpoint",
                "runs_dir", "train_variant", "pooling"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    for key in ("seed", "seq_len", "batch_size", "epochs", "lr"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if getattr(args, "enforced", False):
        config["enforced"] = True
    if getattr(args, "random_smiles", False):
        config["random_smiles"] = True
    if getattr(args, "full_data", False):
        config["full_data"] = True
    if args.full_eval:
        config["full_eval_splits"] = ["valid", "test"]
    elif args.no_full_eval:
        config["full_eval_splits"] = []
    config["repo_root"] = str(REPO_ROOT)
    config["scaffold_root_path"] = str((REPO_ROOT / config["scaffold_root"]).resolve())
    checkpoint_text = str(config.get("pretrained_checkpoint") or "").strip()
    config["pretrained_checkpoint_path"] = (
        str((REPO_ROOT / checkpoint_text).resolve()) if checkpoint_text else ""
    )
    config["runs_dir_path"] = str((REPO_ROOT / config["runs_dir"]).resolve())
    return config

def stage_env(ctx: Dict[str, Any]) -> None:
    info: Dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "modules": {},
    }
    required = ["numpy", "pandas", "torch", "rdkit", "sklearn", "matplotlib", "scipy"]
    optional = ["openpyxl", "tqdm", "ijson"]
    missing = []
    for name in required + optional:
        try:
            module = importlib.import_module(name)
            info["modules"][name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            info["modules"][name] = f"MISSING ({exc})"
            if name in required:
                missing.append(name)
    if missing:
        raise PipelineError(f"missing required packages: {', '.join(missing)}")

    import torch

    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    info["torch_cuda"] = torch.version.cuda
    write_json(ctx["run_dir"] / "01_env.json", info)
    log(f"env: python={sys.version.split()[0]} torch={info['modules']['torch']} cuda={info['cuda_available']}")
    if not info["cuda_available"]:
        log("WARNING: CUDA not available, the pipeline will run on CPU (slower).")

def stage_sanity(ctx: Dict[str, Any]) -> None:
    import numpy as np
    import torch

    from Dataset import BertDataset, build_bert_graph_matrix
    from Tokenizer import BertTokenizer
    from config import BertConfig
    from tool import shortest_path_matrix, sm2graph

    tokenizer = BertTokenizer()
    report: Dict[str, Any] = {"checks": {}, "controlled_groups": []}

    sample = "CC(C)Oc1[nH]ccn1"
    tokens = tokenizer.tokenize(sample)
    ids = tokenizer.sm2index(tokens)
    report["checks"]["tokenizer"] = {
        "smiles": sample,
        "tokens": tokens,
        "token_ids": [int(i) for i in ids],
        "status": "PASS" if tokens and len(tokens) == len(ids) else "FAIL",
    }
    if not tokens or len(tokens) != len(ids):
        raise PipelineError("tokenizer sanity check failed")

    graph = sm2graph(sample)
    dist = np.asarray(shortest_path_matrix(graph), dtype=float)
    symmetric = bool(np.allclose(dist, dist.T))
    zero_diag = bool(np.allclose(np.diag(dist), 0.0))
    report["checks"]["stg"] = {
        "n_atoms": int(dist.shape[0]),
        "distance_shape": list(dist.shape),
        "symmetric": symmetric,
        "zero_diagonal": zero_diag,
        "status": "PASS" if symmetric and zero_diag else "FAIL",
    }
    if not (symmetric and zero_diag):
        raise PipelineError("STG shortest-path matrix sanity check failed")

    config = BertConfig()
    config.seq_len = ctx["config"]["seq_len"]
    config.stg = True
    config.learnable_pos = True
    config.MLM = False
    config.G_GPT = False
    config.reg = False
    config.mutcls = False
    config.tgt_size = 1
    config.z = False
    dataset = BertDataset([["CCO", [0.0]], ["OCC", [0.0]], [sample, [1.0]]], config)
    first_len = len(dataset.input_sm[0])
    stg_shape = list(np.asarray(dataset.s_g[0]).shape)
    report["checks"]["preprocessing"] = {
        "n_records": len(dataset.input_sm),
        "padded_length": first_len,
        "seq_len": ctx["config"]["seq_len"],
        "stg_matrix_shape": stg_shape,
        "cls_prefix": int(dataset.input_sm[0][0]),
        "status": "PASS" if first_len == ctx["config"]["seq_len"] else "FAIL",
    }
    if first_len != ctx["config"]["seq_len"]:
        raise PipelineError("real preprocessing did not pad to config.seq_len")

    write_json(ctx["run_dir"] / "02_code_sanity.json", report)
    log("sanity: tokenizer=PASS STG=PASS preprocessing=PASS")

def stage_data(ctx: Dict[str, Any]) -> None:
    config = ctx["config"]
    run_dir = ctx["run_dir"]
    scaffold_root = Path(config["scaffold_root_path"])
    task = config["task"]

    task_dir = find_task_dir(scaffold_root, task)
    if not task_dir.is_dir():
        raise PipelineError(f"task directory not found under {scaffold_root}: {task}")

    seed = int(config["seed"])
    seed_dir = task_dir / f"seed{seed}"
    if has_split_file(seed_dir, "train") or has_aggregate_csv(seed_dir):
        split_dir = seed_dir
        layout = "per-seed splits"
    elif has_split_file(task_dir, "train") or has_aggregate_csv(task_dir):
        split_dir = task_dir
        layout = "single split reused for every seed"
    else:
        candidates = sorted(
            p for p in task_dir.glob("seed*")
            if has_split_file(p, "train") or has_aggregate_csv(p)
        )
        if not candidates:
            raise PipelineError(f"no split files for task={task} under {task_dir}")
        split_dir = candidates[0]
        layout = f"fallback: {split_dir.name}"
        log(f"data: WARNING split for seed{seed} not found in {task}; using {split_dir.name}")

    split_info: Dict[str, Any] = {
        "task_dir": str(task_dir),
        "split_dir": str(split_dir),
        "seed": seed,
        "layout": layout,
        "splits": {},
    }
    rows_by_split: Dict[str, List[Any]] = {}
    selected_paths: Dict[str, str] = {}
    for name in ("train", "valid", "test"):
        try:
            rows, path = read_split(
                split_dir,
                name,
                enforced=bool(config.get("enforced")),
                random_smiles=bool(config.get("random_smiles")) and name == "train",
            )
        except FileNotFoundError as exc:
            raise PipelineError(str(exc)) from exc
        rows_by_split[name] = rows
        selected_paths[name] = str(path)
        split_info["splits"][name] = {"path": str(path), "n_records": len(rows)}
    if (split_dir / "manifest.json").is_file():
        split_info["manifest"] = read_json(split_dir / "manifest.json", {})

    smoke_info = {}
    if not config.get("full_data", False):
        smoke_root = run_dir / "data" / config["smoke_task"]
        smoke_root.mkdir(parents=True, exist_ok=True)
        for name, size in config["smoke_sizes"].items():
            rows = rows_by_split[name][: int(size)]
            target = smoke_root / f"{name}.csv"
            with open(target, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["smiles", "tags"])
                for row in rows:
                    writer.writerow([row[0], json.dumps(row[1], separators=(",", ":"))])
            smoke_info[name] = {"path": str(target), "n_records": len(rows)}
        if (split_dir / "manifest.json").is_file():
            import shutil

            shutil.copyfile(split_dir / "manifest.json", smoke_root / "manifest.json")
        split_info["smoke_subset"] = smoke_info
    else:
        split_info["full_data"] = True
    split_info["selected_paths"] = selected_paths
    split_info["random_smiles"] = bool(config.get("random_smiles"))
    split_info["enforced"] = bool(config.get("enforced"))

    payload = {
        "scaffold_root": str(scaffold_root),
        "task": task,
        "splits": split_info,
    }
    write_json(run_dir / "03_data.json", payload)
    log(f"data: seed={seed} layout={layout} split={split_dir.name}")
    log(f"data: train={split_info['splits']['train']['n_records']} "
        f"valid={split_info['splits']['valid']['n_records']} "
        f"test={split_info['splits']['test']['n_records']} "
        f"mode={'full' if config.get('full_data', False) else 'smoke'}")

def stage_train(ctx: Dict[str, Any]) -> None:
    config = ctx["config"]
    run_dir = ctx["run_dir"]
    variant = config["train_variant"]
    out_dir = run_dir / "train"
    pretrained_text = config["pretrained_checkpoint_path"]
    pretrained = Path(pretrained_text) if pretrained_text else None
    command = [
        sys.executable, str(PIPELINE_DIR / "train_variant.py"),
        "--mode", "train",
        "--variant", variant,
        "--scaffold_root", str(config["scaffold_root_path"] if config.get("full_data", False)
                                else run_dir / "data"),
        "--task", config["task"] if config.get("full_data", False) else config["smoke_task"],
        "--out_dir", str(out_dir),
        "--epochs", config["epochs"],
        "--seq_len", config["seq_len"],
        "--batch_size", config["batch_size"],
        "--lr", config["lr"],
        "--pooling", config["pooling"],
        "--seed", config["seed"],
    ]
    if pretrained is not None and pretrained.is_file():
        command += ["--pretrained", str(pretrained)]
        log(f"train: initialising from {pretrained}")
    elif pretrained is not None:
        log(f"train: WARNING pretrained checkpoint not found ({pretrained}); training from scratch")
    else:
        log("train: no pretrained checkpoint configured; training from scratch")
    if config.get("enforced"):
        command.append("--enforced")
    if config.get("random_smiles"):
        command.append("--random-smiles")
    run_command(command, cwd=out_dir, log_file=run_dir / "logs" / "train.log")

    result = read_json(out_dir / f"train_{variant.replace('+', 'p')}.json")
    if not result:
        raise PipelineError(f"training did not produce a result file in {out_dir}")
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_file():
        raise PipelineError(f"training did not save a checkpoint at {checkpoint}")
    numeric = {}
    for key, value in (result.get("details", {}) or {}).items():
        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            continue
    if not numeric or any(value != value for value in numeric.values()):
        raise PipelineError(f"training metrics are not finite: {result.get('details')}")
    summary = " ".join(f"{key}={numeric[key]:.4f}"
                       for key in ("AUC", "loss") if key in numeric)
    log(f"train: variant={variant} split={result['split']} {summary} -> {checkpoint}")

def _eval_rows(ctx: Dict[str, Any], variant: str, checkpoint: Path, split: str,
               out_dir: Path) -> Optional[Dict[str, Any]]:
    config = ctx["config"]
    if not checkpoint.is_file():
        log(f"eval: SKIP {variant} ({split}) - checkpoint not found: {checkpoint}")
        return None
    full_scale = config.get("full_data", False) or split in config["full_eval_splits"]
    task = config["task"] if full_scale else config["smoke_task"]
    scaffold_root = config["scaffold_root_path"] if full_scale else str(ctx["run_dir"] / "data")
    command = [
        sys.executable, str(PIPELINE_DIR / "train_variant.py"),
        "--mode", "eval",
        "--variant", variant,
        "--scaffold_root", scaffold_root,
        "--task", task,
        "--split", split,
        "--checkpoint", str(checkpoint),
        "--out_dir", str(out_dir),
        "--seq_len", config["seq_len"],
        "--batch_size", config["batch_size"],
        "--pooling", config["pooling"],
        "--seed", config["seed"],
    ]
    if config.get("enforced"):
        command.append("--enforced")
    if config.get("random_smiles"):
        command.append("--random-smiles")
    log_file = ctx["run_dir"] / "logs" / f"eval_{variant.replace('+', 'p')}_{split}_{task}.log"
    run_command(command, cwd=out_dir, log_file=log_file)
    result = read_json(out_dir / f"eval_{variant.replace('+', 'p')}.json")
    if not result:
        raise PipelineError(f"evaluation produced no result for {variant} on {split}")
    details = result.get("details", {})
    parts = " ".join(f"{key}={float(details[key]):.4f}"
                     for key in ("AUC", "loss")
                     if details.get(key) is not None)
    log(f"eval: {variant:7s} split={split:5s} task={task:16s} {parts} ckpt={checkpoint}")
    return result

def stage_eval(ctx: Dict[str, Any]) -> None:
    config = ctx["config"]
    run_dir = ctx["run_dir"]
    summaries: List[Dict[str, Any]] = []

    train_result = read_json(run_dir / "train" / f"train_{config['train_variant'].replace('+', 'p')}.json")
    if train_result:
        result = _eval_rows(
            ctx, config["train_variant"], Path(train_result["checkpoint"]), "test",
            run_dir / "eval" / "trained" / config["train_variant"].replace("+", "p"),
        )
        if result:
            result["source"] = "trained_in_pipeline"
            summaries.append(result)

    write_json(run_dir / "05_eval.json", summaries)
    if not summaries:
        raise PipelineError("no evaluation ran; nothing to report")

def stage_report(ctx: Dict[str, Any]) -> None:
    config = ctx["config"]
    run_dir = ctx["run_dir"]

    env_info = read_json(run_dir / "01_env.json", {}) or {}
    sanity = read_json(run_dir / "02_code_sanity.json", {}) or {}
    data = read_json(run_dir / "03_data.json", {}) or {}
    evals = read_json(run_dir / "05_eval.json", []) or []

    lines: List[str] = []
    lines.append("# STGFormer pipeline run report")
    lines.append("")
    lines.append(f"- run directory: `{run_dir}`")
    lines.append(f"- started: {ctx['started']}")
    lines.append(f"- elapsed: {int(time.time() - ctx['t0'])} s")
    lines.append(f"- python: `{env_info.get('executable')}`")
    lines.append(f"- torch: {env_info.get('modules', {}).get('torch')} "
                 f"cuda={env_info.get('cuda_available')} ({env_info.get('cuda_device_name')})")
    lines.append("")

    lines.append("## 1. Real-code sanity checks")
    lines.append("")
    for name, payload in (sanity.get("checks", {}) or {}).items():
        status = payload.get("status", payload.get("all_pass"))
        lines.append(f"- `{name}`: {status}")
    lines.append("")

    lines.append("## 2. Data")
    lines.append("")
    for name in ("train", "valid", "test"):
        entry = ((data.get("splits", {}) or {}).get("splits", {}) or {}).get(name)
        if entry:
            lines.append(f"- {name}: {entry['n_records']} records (`{entry['path']}`)")
    lines.append("")

    lines.append("## 3. Training (project STGFormerTrainer, no re-implementation)")
    lines.append("")
    train_result = read_json(run_dir / "train" / f"train_{config['train_variant'].replace('+', 'p')}.json")
    if train_result:
        details = train_result.get("details", {})
        lines.append(f"- variant: {train_result['variant']} "
                     f"(STG={train_result['stg']}, "
                     f"learnable_pos={train_result['learnable_pos']})")
        lines.append(f"- initialised from: `{train_result.get('pretrained')}`")
        lines.append(f"- checkpoint written: `{train_result.get('checkpoint')}`")
        lines.append(f"- validation AUC: {details.get('AUC')}")
    else:
        lines.append("- (no training result in this run)")
    lines.append("")

    lines.append("## 4. Evaluation (actual checkpoints printed)")
    lines.append("")
    lines.append("| source | variant | split | task | AUC | checkpoint |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in evals:
        details = item.get("details", {})

        def _fmt(value: Any) -> str:
            return "-" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {item.get('source')} | {item.get('variant')} | {item.get('split')} | "
            f"{item.get('task')} | {_fmt(details.get('AUC'))} | `{item.get('checkpoint')}` |"
        )
    lines.append("")

    lines.append("## 5. Artefacts")
    lines.append("")
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".txt", ".png", ".svg", ".pdf"}:
            lines.append(f"- `{path.relative_to(run_dir)}`")
    lines.append("")

    report_path = run_dir / "run_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary_path = run_dir / "metrics_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        for item in evals:
            details = item.get("details", {})
            label = f"{item.get('source')}|{item.get('variant')}|{item.get('split')}|{item.get('task')}"
            for metric in ("AUC", "loss"):
                if details.get(metric) is not None:
                    writer.writerow(["eval", f"{label}|{metric}", details[metric]])
    write_json(run_dir / "run_manifest.json", {
        "started": ctx["started"],
        "stages": ctx["stages_requested"],
        "config": config,
        "run_dir": str(run_dir),
        "artifacts": {"report": str(report_path), "metrics_summary": str(summary_path)},
    })
    log(f"report: {report_path}")

STAGES = {
    "env": stage_env,
    "sanity": stage_sanity,
    "data": stage_data,
    "train": stage_train,
    "eval": stage_eval,
    "report": stage_report,
}

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STGFormer model pipeline.")
    parser.add_argument("--stages", default="all",
                        help="comma separated subset of: " + ",".join(STAGE_ORDER) + " (or 'all')")
    parser.add_argument("--run-dir", default="", help="reuse/create a specific run directory")
    parser.add_argument("--config", default="", help="path to a pipeline config JSON")
    parser.add_argument("--task", choices=PAPER_TASKS, default=None)
    parser.add_argument("--smoke-task", default=None)
    parser.add_argument("--scaffold-root", default=None)
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--enforced", action="store_true",
                        help="use enforced CSV data for train/valid/test when available")
    parser.add_argument("--random-smiles", action="store_true",
                        help="use enforced/random-SMILES training data only")
    parser.add_argument("--full-data", action="store_true",
                        help="train and evaluate on the complete selected dataset (no smoke subset)")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--train-variant", default=None)
    parser.add_argument("--pooling", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--full-eval", action="store_true",
                        help="evaluate the trained checkpoint on the full test split")
    parser.add_argument("--no-full-eval", action="store_true",
                        help="evaluate only the run-local smoke test subset")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = resolve_config(args)

    if args.stages.strip().lower() in {"all", ""}:
        stages = list(STAGE_ORDER)
    else:
        stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    unknown = [item for item in stages if item not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; expected {STAGE_ORDER}")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = Path(config["runs_dir_path"]) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx: Dict[str, Any] = {
        "config": config,
        "run_dir": run_dir,
        "stages_requested": stages,
        "started": datetime.now().isoformat(timespec="seconds"),
        "t0": time.time(),
    }

    print("=" * 78)
    print("STGFormer model pipeline")
    print("=" * 78)
    log(f"run directory: {run_dir}")
    log(f"stages: {', '.join(stages)}")

    for name in stages:
        log(f"--- stage: {name} ---")
        started = time.time()
        STAGES[name](ctx)
        log(f"--- stage {name} done in {time.time() - started:.1f}s ---")

    evals = read_json(run_dir / "05_eval.json", []) or []
    print()
    print("=" * 78)
    print("PIPELINE DONE")
    print("=" * 78)
    print(f"run directory : {run_dir}")
    print(f"report        : {run_dir / 'run_report.md'}")
    print(f"metrics       : {run_dir / 'metrics_summary.csv'}")
    for item in evals:
        details = item.get("details", {})
        auc = details.get("AUC")
        print(f"  {item.get('source', '?'):20s} {item.get('variant', '?'):7s} "
              f"{item.get('split', '?'):5s} {item.get('task', '?'):16s} "
              f"AUC={'n/a' if auc is None else format(float(auc), '.4f')}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        print(f"\nPIPELINE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
