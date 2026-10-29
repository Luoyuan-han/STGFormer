from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import BertConfig
from Dataset import build_bert_graph_matrix, enc_input_padding
from Tokenizer import BertTokenizer
from modell import BertModel

THRESHOLDS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3)
VARIANTS = {
    "base": {"stg": False, "learnable_pos": False},
    "pe": {"stg": False, "learnable_pos": True},
    "stg": {"stg": True, "learnable_pos": False},
    "stg+pe": {"stg": True, "learnable_pos": True},
}


def default_input(dataset: str, task: str, split: str) -> Path:
    if dataset == "qm9":
        return REPO_ROOT / "fintune_datas" / "compas_qm9" / "qm9_gap" / f"{split}_meta.json"
    task_name = "compas_homo" if task.lower() in {"homo", "homo_ev"} else "compas_lumo"
    return REPO_ROOT / "fintune_datas" / "compas_qm9" / task_name / f"{split}_meta.json"


def parse_label(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        output = []
        for index, row in enumerate(rows):
            smiles = row.get("smiles") or row.get("SMILES") or row.get("Smiles")
            if not smiles:
                continue
            molecule_id = row.get("molecule_id") or row.get("mol_id") or row.get("id") or str(index)
            label = row.get("target") or row.get("label") or row.get("tags")
            output.append({"index": index, "molecule_id": molecule_id, "smiles": smiles, "target": parse_label(label)})
        return output
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("data", []))
    output = []
    for index, row in enumerate(payload):
        if isinstance(row, dict):
            smiles = row.get("smiles") or row.get("SMILES")
            target = row.get("target", row.get("label"))
            meta = row.get("meta") or {}
            molecule_id = meta.get("molecule") or meta.get("mol_id") or row.get("molecule_id") or str(index)
        else:
            smiles = row[0] if row else None
            target = row[1] if len(row) > 1 else None
            molecule_id = str(index)
        if smiles:
            output.append({"index": index, "molecule_id": molecule_id, "smiles": str(smiles), "target": parse_label(target)})
    return output


def variant_from_checkpoint(path: Path) -> str:
    """Infer the architecture variant from the checkpoint path."""
    name = "/".join(part.lower() for part in path.parts)
    if "stg+pe" in name or "stg_pe" in name or "stgpe" in name:
        return "stg+pe"
    if "stg" in name:
        return "stg"
    if "rope" in name:
        return "base"
    if "pe" in name or "learnable" in name:
        return "pe"
    return "base"


def infer_stage(state: dict[str, torch.Tensor]) -> str:
    if any(key == "Encoder.fc_out.weight" for key in state):
        return "finetuned"
    if any(key == "Encoder.tokens.weight" for key in state):
        return "pretrained"
    raise RuntimeError("checkpoint does not contain Encoder.fc_out or Encoder.tokens; unsupported checkpoint format")


def infer_seq_len(state: dict[str, torch.Tensor], fallback: int) -> int:
    position = state.get("Encoder.pos_emb")
    if position is not None and position.ndim == 3:
        return int(position.shape[1])
    return int(fallback)


def infer_target_size(state: dict[str, torch.Tensor]) -> int:
    for key in ("Encoder.fc_out.weight", "Encoder.tokens.weight"):
        weight = state.get(key)
        if weight is not None and weight.ndim == 2:
            return int(weight.shape[0])
    raise RuntimeError("checkpoint output head has no usable weight tensor")


def make_config(variant: str, seq_len: int, stage: str, target_size: int, pooling: str) -> BertConfig:
    spec = VARIANTS[variant]
    pretrained = stage == "pretrained"
    return BertConfig(seq_len=seq_len, stg=spec["stg"], learnable_pos=spec["learnable_pos"], rope=False,
                      MLM=pretrained, G_GPT=False, reg=False, mutcls=pretrained or target_size > 1,
                      tgt_size=target_size, vocab_size=156, pooling=pooling, use_solvent=False)


def load_model(checkpoint: Path, variant: str, stage: str, seq_len: int,
               pooling: str, device: torch.device) -> tuple[BertModel, BertConfig, str, int]:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (PermissionError, OSError) as exc:
        raise RuntimeError(
            f"cannot read checkpoint '{checkpoint}'. Windows reported '{exc}'. "
            "Check that the drive is online and the file is not locked, then copy the checkpoint "
            "to a local NTFS path and rerun with that path."
        ) from exc
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    detected_stage = infer_stage(state)
    if stage == "auto":
        stage = detected_stage
    elif stage != detected_stage:
        raise RuntimeError(f"checkpoint is {detected_stage}, but --stage {stage} was requested")
    seq_len = infer_seq_len(state, seq_len)
    target_size = infer_target_size(state)
    config = make_config(variant, seq_len, stage, target_size, pooling)
    model = BertModel(config)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(key for key in expected.keys() & state.keys() if expected[key].shape != state[key].shape)
    if missing or unexpected or mismatched:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}, shape={mismatched}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config, stage, target_size


def molecule_metadata(record: dict[str, Any], tokenizer: BertTokenizer, config: BertConfig) -> dict[str, Any] | None:
    tokens = tokenizer.tokenize(record["smiles"])
    solute_len = len(tokens) if tokens is not None else 0
    if not tokens:
        return None
    ids = tokenizer.sm2index(tokens)
    if len(ids) > config.seq_len - 1:
        raise ValueError(f"{record['smiles']} has {len(ids)} tokens but seq_len={config.seq_len}")
    mol = Chem.MolFromSmiles(record["smiles"])
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    signature = tuple(sorted(Counter(int(value) for value in ids if int(value) not in {0, 1, 2, 3}).items()))
    return {
        **record,
        "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "n_atoms": mol.GetNumAtoms(),
        "n_bonds": mol.GetNumBonds(),
        "token_count": len(ids),
        "token_multiset": json.dumps(signature, separators=(",", ":")),
        "token_text": " ".join(tokens),
        "token_ids": json.dumps(ids, separators=(",", ":")),
        "tokens": tokens,
        "solute_len": solute_len,
    }


class RepresentationDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], config: BertConfig):
        self.items = []
        tokenizer = BertTokenizer()
        for record in records:
            item = molecule_metadata(record, tokenizer, config)
            if item is None:
                continue
            ids = tokenizer.sm2index(item["tokens"])
            graph = build_bert_graph_matrix(item["tokens"], item["solute_len"], config)
            graph = F.pad(graph, [1, config.seq_len - len(ids) - 1] * 2, value=0)
            graph = graph.expand(config.num_heads, config.seq_len, config.seq_len).clone()
            inputs = torch.tensor(enc_input_padding(ids[:], config.seq_len), dtype=torch.long)
            self.items.append((item, inputs, torch.tensor(len(ids), dtype=torch.long), graph))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def collate(batch):
    records, inputs, lengths, graphs = zip(*batch)
    return list(records), torch.stack(inputs), torch.stack(lengths), torch.stack(graphs)


@torch.inference_mode()
def extract_embeddings(model: BertModel, dataset: RepresentationDataset, device: torch.device,
                       batch_size: int, stage: str) -> tuple[np.ndarray, np.ndarray | None]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0]

    handle = model.Encoder.layers[-1].register_forward_hook(hook) if stage == "pretrained" else None
    chunks = []
    prediction_chunks = []
    try:
        for _records, inputs, lengths, graphs in loader:
            outputs = model(inputs.to(device), lengths.to(device), graphs.to(device))
            if stage == "pretrained":
                chunks.append(captured.pop("hidden")[:, 0, :].cpu().numpy().astype(np.float32))
            else:
                pooled = outputs[1]
                if pooled is None:
                    raise RuntimeError("fine-tuned checkpoint did not return a pooled encoder representation")
                chunks.append(pooled[:, 0, :].cpu().numpy().astype(np.float32))
                prediction_chunks.append(outputs[0][:, 0, :].cpu().numpy().astype(np.float32))
    finally:
        if handle is not None:
            handle.remove()
    embeddings = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 768), dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise RuntimeError("non-finite representation detected")
    predictions = np.concatenate(prediction_chunks, axis=0) if prediction_chunks else None
    return embeddings, predictions


def controlled_groups(records: list[dict[str, Any]], group_key: str) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        key = record["token_multiset"] if group_key == "tokens" else f"{record['formula']}|{record['token_multiset']}"
        groups.setdefault(key, []).append(index)
    return [(key, indices) for key, indices in groups.items()
            if len(indices) > 1 and len({records[index]["canonical_smiles"] for index in indices}) > 1]


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0:
        return float("nan")
    return float(np.clip(1.0 - np.dot(left, right) / denominator, 0.0, 2.0))


def calculate_pairs(records: list[dict[str, Any]], embeddings: np.ndarray, group_key: str,
                    predictions: np.ndarray | None = None):
    groups = controlled_groups(records, group_key)
    pairs = []
    group_rows = []
    for group_id, (key, indices) in enumerate(groups):
        group_rows.append({"group_id": group_id, "group_key": key, "n_molecules": len(indices),
                           "molecule_indices": json.dumps(indices)})
        for left, right in combinations(indices, 2):
            if records[left]["canonical_smiles"] == records[right]["canonical_smiles"]:
                continue
            row = {"group_id": group_id, "group_key": key,
                          "molecule_i": records[left]["molecule_id"], "molecule_j": records[right]["molecule_id"],
                          "smiles_i": records[left]["smiles"], "smiles_j": records[right]["smiles"],
                          "target_i": records[left]["target"], "target_j": records[right]["target"],
                          "cosine_distance": cosine_distance(embeddings[left], embeddings[right])}
            if predictions is not None:
                target_i = float(records[left]["target"])
                target_j = float(records[right]["target"])
                pred_i = float(predictions[left, 0])
                pred_j = float(predictions[right, 0])
                if all(np.isfinite(value) for value in (target_i, target_j, pred_i, pred_j)):
                    true_difference = abs(target_i - target_j)
                    predicted_difference = abs(pred_i - pred_j)
                    row.update({
                        "prediction_i": pred_i,
                        "prediction_j": pred_j,
                        "true_pair_difference": true_difference,
                        "predicted_pair_difference": predicted_difference,
                        "pair_property_error": abs(true_difference - predicted_difference),
                    })
            pairs.append(row)
    return pairs, group_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_statistics(records: list[dict[str, Any]]) -> tuple[float, float]:
    values = np.asarray([float(record["target"]) for record in records], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot calculate target statistics: no finite targets in training records")
    mean = float(values.mean())
    std = float(values.std())
    return mean, std if std >= 1e-12 else 1.0


def summarize(pairs, groups, records, checkpoint: Path, variant: str) -> dict[str, Any]:
    values = np.asarray([row["cosine_distance"] for row in pairs], dtype=float)
    summary = {"checkpoint": str(checkpoint), "variant": variant, "n_records": len(records),
               "n_controlled_groups": len(groups), "n_pairs": int(values.size),
               "mean_cosine_distance": float(np.mean(values)) if values.size else None,
               "median_cosine_distance": float(np.median(values)) if values.size else None}
    for threshold in THRESHOLDS:
        summary[f"fraction_below_{threshold:.0e}"] = float(np.mean(values < threshold)) if values.size else None
    pair_errors = np.asarray([row["pair_property_error"] for row in pairs if "pair_property_error" in row], dtype=float)
    true_differences = np.asarray([row["true_pair_difference"] for row in pairs if "true_pair_difference" in row], dtype=float)
    predicted_differences = np.asarray([row["predicted_pair_difference"] for row in pairs if "predicted_pair_difference" in row], dtype=float)
    summary.update({
        "n_property_pairs": int(pair_errors.size),
        "mean_true_pair_difference": float(np.mean(true_differences)) if true_differences.size else None,
        "mean_predicted_pair_difference": float(np.mean(predicted_differences)) if predicted_differences.size else None,
        "delta_mae": float(np.mean(pair_errors)) if pair_errors.size else None,
        "pair_property_error_sd": float(np.std(pair_errors, ddof=1)) if pair_errors.size > 1 else None,
    })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract STGFormer representations and controlled-pair cosine distances.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["compas", "qm9"], default="qm9")
    parser.add_argument("--task", choices=["homo", "lumo", "gap"], default="gap")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--input", default="")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="")
    parser.add_argument("--metric", choices=["extract", "cosine", "both"], default="both")
    parser.add_argument("--group-key", choices=["tokens", "formula_tokens"], default="formula_tokens")
    parser.add_argument("--stage", choices=["auto", "pretrained", "finetuned"], default="auto",
                        help="checkpoint stage; auto detects MLM (pretrained) versus regression (fine-tuned)")
    parser.add_argument("--seq-len", type=int, default=0,
                        help="input length; 0 reads it from Encoder.pos_emb, otherwise uses 128")
    parser.add_argument("--pooling", choices=["auto", "cls", "mean"], default="auto",
                        help="fine-tuned readout pooling; auto uses the paper's CLS readout")
    parser.add_argument("--prediction-scale", choices=["auto", "raw", "standardized"], default="auto",
                        help="scale of fine-tuned regression outputs; auto follows the paper and inverse-transforms z-scores")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "qm9" and args.task != "gap":
        raise ValueError("QM9 analysis requires --task gap")
    if args.dataset == "compas" and args.task not in {"homo", "lumo"}:
        raise ValueError("COMPAS analysis requires --task homo or --task lumo")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    input_path = Path(args.input).expanduser() if args.input else default_input(args.dataset, args.task, args.split)
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    variant = args.variant.lower() or variant_from_checkpoint(checkpoint)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output = Path(args.output).expanduser() if args.output else REPO_ROOT / "runs" / f"analysis_{args.dataset}_{args.task}_{variant}"
    output = output if output.is_absolute() else (REPO_ROOT / output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = read_records(input_path)
    if args.limit > 0:
        records = records[:args.limit]
    model_stage = args.stage
    requested_seq_len = args.seq_len if args.seq_len > 0 else 128
    requested_pooling = args.pooling
    if requested_pooling == "auto":
        requested_pooling = "cls"
    model, config, model_stage, target_size = load_model(
        checkpoint, variant, model_stage, requested_seq_len, requested_pooling, device
    )
    dataset = RepresentationDataset(records, config)
    usable_records = [item[0] for item in dataset.items]
    embeddings, predictions = extract_embeddings(model, dataset, device, args.batch_size, model_stage)
    prediction_scale = "none"
    target_mean = None
    target_std = None
    if predictions is not None and args.metric in {"cosine", "both"}:
        prediction_scale = args.prediction_scale
        if prediction_scale == "auto":
            prediction_scale = "standardized"
        if prediction_scale == "standardized":
            if input_path.name == f"{args.split}_meta.json":
                train_path = input_path.with_name("train_meta.json")
            elif input_path.name == f"{args.split}.csv":
                train_path = input_path.with_name("train.csv")
            else:
                train_path = input_path.with_name(input_path.name.replace(args.split, "train", 1))
            if not train_path.is_file():
                raise FileNotFoundError(
                    f"training split is required to inverse-transform standardized predictions: {train_path}"
                )
            target_mean, target_std = target_statistics(read_records(train_path))
            predictions = predictions * target_std + target_mean
    elif predictions is not None:
        predictions = None
    molecule_rows = []
    for index, record in enumerate(usable_records):
        molecule_rows.append({key: value for key, value in record.items() if key not in {"tokens", "solute_len"}} | {"embedding_index": index})
    write_csv(output / "molecules.csv", molecule_rows)
    np.save(output / "embeddings.npy", embeddings)

    result = {"dataset": args.dataset, "task": args.task, "split": args.split, "input": str(input_path),
              "checkpoint": str(checkpoint), "variant": variant, "stage": model_stage,
              "target_size": target_size, "pooling": config.pooling, "prediction_scale": prediction_scale,
              "target_mean": target_mean, "target_std": target_std, "metric": args.metric,
              "group_key": args.group_key, "seq_len": config.seq_len, "batch_size": args.batch_size,
              "device": str(device),
              "n_input_records": len(records), "n_usable_records": len(usable_records)}
    if args.metric in {"cosine", "both"}:
        pairs, groups = calculate_pairs(usable_records, embeddings, args.group_key, predictions)
        write_csv(output / "controlled_groups.csv", groups)
        write_csv(output / "cosine_pairs.csv", pairs)
        if predictions is not None:
            write_csv(output / "pair_property_errors.csv", pairs)
        controlled_indices = sorted({int(index) for group in groups for index in json.loads(group["molecule_indices"])})
        write_csv(output / "controlled_molecules.csv", [molecule_rows[index] | {"group_membership": True} for index in controlled_indices])
        result.update(summarize(pairs, groups, usable_records, checkpoint, variant))
    else:
        result.update({"n_controlled_groups": None, "n_pairs": None})
    (output / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
