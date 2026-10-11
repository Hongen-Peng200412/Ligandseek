from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXCLUSION_RESNAMES = {
    "HOH", "WAT", "H2O", "DOD",
    "GOL", "EDO", "MPD", "PEG", "PG4", "P6G",
    "TRS", "MES", "HEP", "ACT", "CIT",
    "EOH", "MOH", "IPA", "DMS", "DTT", "BME",
    "BU1", "TBU",
    "UNX", "UNL", "UNK",
    "XE", "KR",
}


@dataclass(frozen=True)
class LigandRule:
    class_id: int
    class_name: str
    conditions: dict[str, Any] = field(default_factory=dict)
    binding_threshold: float | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LigandRule":
        conditions = dict(data.get("conditions") or {})
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            conditions=conditions,
            binding_threshold=(
                None
                if data.get("binding_threshold") is None
                else float(data["binding_threshold"])
            ),
        )

    def accepts(self, record: dict[str, Any]) -> bool:
        return all(_condition_accepts(record.get(field_name), condition) for field_name, condition in self.conditions.items())


@dataclass(frozen=True)
class LigandRuleConfig:
    rules: tuple[LigandRule, ...]
    exclusion_resnames: frozenset[str] = field(default_factory=frozenset)
    use_exclusion_resnames: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LigandRuleConfig":
        exclusion = data.get("exclusion_resnames")
        if exclusion is None:
            exclusion_set = set(DEFAULT_EXCLUSION_RESNAMES)
        else:
            exclusion_set = {str(value).strip().upper() for value in exclusion}

        return cls(
            rules=tuple(LigandRule.from_mapping(item) for item in data.get("rules", [])),
            exclusion_resnames=frozenset(exclusion_set),
            use_exclusion_resnames=bool(data.get("use_exclusion_resnames", True)),
        )


def load_rule_config(path: str | Path) -> LigandRuleConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return LigandRuleConfig.from_mapping(data)


def apply_ligand_rules(
    records: list[dict[str, Any]],
    config: LigandRuleConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for row_index, record in enumerate(records):
        resname = str(record.get("resname", "")).strip().upper()
        if config.use_exclusion_resnames and resname in config.exclusion_resnames:
            rejected.append({**record, "row_index": row_index, "status": "excluded_resname"})
            continue

        matched_rule = next((rule for rule in config.rules if rule.accepts(record)), None)
        if matched_rule is None:
            rejected.append({**record, "row_index": row_index, "status": "no_rule"})
            continue

        output = {
            **record,
            "row_index": row_index,
            "class_id": matched_rule.class_id,
            "class_name": matched_rule.class_name,
        }
        if matched_rule.binding_threshold is not None:
            output["binding_threshold"] = matched_rule.binding_threshold
        selected.append(output)

    return selected, rejected


def _condition_accepts(value: Any, condition: Any) -> bool:
    if condition is None:
        return True
    if isinstance(condition, dict):
        if "equals" in condition and not _equals(value, condition["equals"]):
            return False
        if "not_equals" in condition and _equals(value, condition["not_equals"]):
            return False
        if "in" in condition and value not in condition["in"]:
            return False
        if "min" in condition and _to_float(value) < float(condition["min"]):
            return False
        if "max" in condition and _to_float(value) > float(condition["max"]):
            return False
        if "range" in condition:
            lower, upper = condition["range"]
            numeric = _to_float(value)
            if numeric < float(lower) or numeric > float(upper):
                return False
        return True
    return _equals(value, condition)


def _equals(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return _to_bool(value) is expected
    return value == expected


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)
