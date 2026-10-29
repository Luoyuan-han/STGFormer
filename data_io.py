from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _as_float(value: Any) -> float:
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return float("nan")
    return float(value)

def _parse_label(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [_as_float(item) for item in value]
    text = str(value).strip()
    if text.startswith(("[", "(")) and text.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [_as_float(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
    if ";" in text:
        return [_as_float(item) for item in text.split(";")]
    return [_as_float(value)]

def _normalise_row(row: Any) -> List[Any]:
    if isinstance(row, dict):
        smiles = row.get("smiles", row.get("SMILES", row.get("Smiles", "")))
        label = row.get("tags", row.get("label", row.get("labels", row.get("y"))))
        if label is None:
            keys = sorted(
                (key for key in row if str(key).lower().startswith("task_")),
                key=lambda key: (
                    0 if str(key).split("_")[-1].isdigit() else 1,
                    int(str(key).split("_")[-1])
                    if str(key).split("_")[-1].isdigit() else str(key),
                ),
            )
            label = [row[key] for key in keys]
        labels = _parse_label(label)
        return [str(smiles), labels]
    if len(row) < 2:
        return []
    smiles, label = row[0], row[1]
    labels = _parse_label(label)
    if len(row) > 2 and isinstance(row[2], dict):
        return [smiles, labels, row[2]]
    return [smiles, labels]

def _split_match(value: str, split: str) -> bool:
    value = str(value).strip().lower()
    split = split.lower()
    aliases = {"valid": {"valid", "validation", "val", "dev"}, "test": {"test"}, "train": {"train"}}
    return value in aliases.get(split, {split})

def read_dataset(path: Path, split: Optional[str] = None) -> List[List[Any]]:

    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"dataset input must be CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        fields = {str(field).strip().lower(): field for field in reader.fieldnames if field}
        split_field = fields.get("split")
        rows: List[List[Any]] = []
        for raw in reader:
            if split and split_field and not _split_match(raw.get(split_field, ""), split):
                continue
            row = _normalise_row(raw)
            if row:
                rows.append(row)
        return rows

def find_task_dir(root: Path, task: str) -> Path:
    candidate = Path(root) / task
    nested = Path(root) / "scaffold" / task
    return candidate if candidate.is_dir() else nested if nested.is_dir() else candidate

def has_split_file(directory: Path, name: str) -> bool:
    return any((directory / f"{name}{suffix}").is_file()
               for suffix in (".csv", "_enforced.csv"))

def has_aggregate_csv(directory: Path) -> bool:
    return any((Path(directory) / filename).is_file()
               for filename in ("dataset_with_split.csv", "dataset_with_split_expanded.csv"))

def choose_split_file(directory: Path, name: str, enforced: bool = False,
                      random_smiles: bool = False) -> Tuple[Path, Optional[str]]:

    use_enforced = enforced or (name == "train" and random_smiles)
    csv_suffixes = ["_enforced.csv", ".csv"] if use_enforced else [".csv"]
    for suffix in csv_suffixes:
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate, None

    aggregate_names = ["dataset_with_split.csv", "dataset_with_split_expanded.csv"]
    for filename in aggregate_names:
        candidate = directory / filename
        if candidate.is_file():
            return candidate, name

    raise FileNotFoundError(f"missing {name} split under {directory} (expected CSV)")

def read_split(directory: Path, name: str, enforced: bool = False,
               random_smiles: bool = False) -> Tuple[List[List[Any]], Path]:
    path, split = choose_split_file(directory, name, enforced, random_smiles)
    return read_dataset(path, split=split), path
