from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

def convert(path: Path, overwrite: bool = False) -> bool:
    target = path.with_suffix(".csv")
    if target.exists() and not overwrite:
        return False
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON array: {path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "tags", "split"])
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            split = row[2] if len(row) > 2 and isinstance(row[2], str) else ""
            writer.writerow([
                row[0],
                json.dumps(row[1], ensure_ascii=False, separators=(",", ":")),
                split,
            ])
    os.replace(temporary, target)
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path,
                        help="dataset roots, e.g. fintune_datas/moleculenet_scaffold fintune_datas/pytdc_scaffold")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing CSV exports")
    parser.add_argument("--delete-json", action="store_true",
                        help="remove each split JSON only after validating its CSV")
    args = parser.parse_args()

    converted = 0
    for root in args.roots:
        for path in sorted(root.rglob("*.json")):
            if path.name in {"manifest.json"} or path.parent.name in {"cache", "manifests"}:
                continue
            if path.stem not in {"train", "valid", "test", "train_enforced", "valid_enforced", "test_enforced"}:
                continue
            csv_path = path.with_suffix(".csv")
            converted += int(convert(path, overwrite=args.overwrite))
            if args.delete_json:
                if not csv_path.is_file() or csv_path.stat().st_size == 0:
                    raise RuntimeError(f"CSV replacement missing or empty: {csv_path}")
                path.unlink()
    print(f"Converted {converted} split JSON files to CSV.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
