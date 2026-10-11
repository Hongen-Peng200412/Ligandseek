from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ligandseek.parse.rule_filter import apply_ligand_rules, load_rule_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ordered ligand-selection rules to an attribute table.")
    parser.add_argument("--input", required=True, help="CSV or JSON list of ligand attribute records.")
    parser.add_argument("--config", default="configs/parse/binary_task.yaml", help="YAML rule configuration.")
    parser.add_argument("--selected-output", required=True, help="Output CSV/JSON path for selected rows.")
    parser.add_argument("--rejected-output", default=None, help="Optional CSV/JSON path for rejected rows.")
    args = parser.parse_args()

    records = _read_records(Path(args.input))
    config = load_rule_config(args.config)
    selected, rejected = apply_ligand_rules(records, config)

    _write_records(Path(args.selected_output), selected)
    if args.rejected_output:
        _write_records(Path(args.rejected_output), rejected)
    print(f"selected={len(selected)} rejected={len(rejected)}")


def _read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [_coerce_row(row) for row in csv.DictReader(handle)]


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _coerce_row(row: dict[str, str]) -> dict:
    return {key: _coerce_value(value) for key, value in row.items()}


def _coerce_value(value: str):
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if text == "":
        return ""
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


if __name__ == "__main__":
    main()
