# src/data/legend.py
from dataclasses import dataclass
from typing import Dict, List
from src.utils.config import load_config

@dataclass(frozen=True)
class Legend:
    class_names: List[str]
    raw_to_class: Dict[int, int]
    class_to_raw: Dict[int, int]
    num_classes: int
    ignore_index: int = 0  # No data

def load_legend(path: str) -> Legend:
    cfg = load_config(path)
    entries = cfg["classes"]

    # se usi raw_id/class_id:
    class_names = [e["name"] for e in entries]
    raw_to_class = {int(e["raw_id"]): int(e["class_id"]) for e in entries}
    class_to_raw = {int(e["class_id"]): int(e["raw_id"]) for e in entries}

    return Legend(
        class_names=class_names,
        raw_to_class=raw_to_class,
        class_to_raw=class_to_raw,
        num_classes=len(class_names),
        ignore_index=0,
    )