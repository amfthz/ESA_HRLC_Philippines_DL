# src/data/indexing.py
import os
import json
from typing import List, Tuple


def list_label_tile_ids(root_labels: str) -> List[str]:
    """
    Canonical list of tile IDs from labels/*.json
    Example: labels/20KPF_12_02.json -> 20KPF_12_02
    """
    tile_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(root_labels)
        if f.endswith(".json") and not f.startswith(".")
    )
    return tile_ids


def build_pairs(
    root_s1: str,
    root_gt: str,
    root_labels: str,
    t_len: int,
) -> List[Tuple[str, List[str]]]:
    """
    Build (GT_path, ordered_S1_paths) pairs using label JSONs as source of truth.
    Any inconsistency → tile skipped.
    """

    pairs = []
    skipped = 0

    json_files = sorted(
        f for f in os.listdir(root_labels)
        if f.endswith(".json") and not f.startswith(".")
    )

    for jf in json_files:
        try:
            tile_id = os.path.splitext(jf)[0]  # e.g. 20KPF_12_02

            gt_name = f"{tile_id.replace('_', '_GT_', 1)}.tif"
            gt_path = os.path.join(root_gt, gt_name)
            if not os.path.isfile(gt_path):
                raise FileNotFoundError("Missing GT")

            with open(os.path.join(root_labels, jf), "r") as f:
                meta = json.load(f)

            s1_list = meta["corresponding_s1"].split(";")
            if len(s1_list) != t_len:
                raise ValueError("Wrong number of S1 features")

            s1_paths = []
            for s1 in s1_list:
                p = os.path.join(root_s1, s1)
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing S1: {s1}")
                s1_paths.append(p)

            pairs.append((gt_path, s1_paths))

        except Exception:
            skipped += 1
            continue

    print(
        f"[INDEXING] JSON files: {len(json_files)} | "
        f"valid samples: {len(pairs)} | skipped: {skipped}"
    )

    return pairs


if __name__ == "__main__":
    root_s1 = '/Volumes/PortableSSD/ESA_CCI_UNIPV/DATASET_LC_MAPPING_JSTARS/training/africa/s1'
    root_gt = '/Volumes/PortableSSD/ESA_CCI_UNIPV/DATASET_LC_MAPPING_JSTARS/training/africa/ground_reference'
    root_labels = '/Volumes/PortableSSD/ESA_CCI_UNIPV/DATASET_LC_MAPPING_JSTARS/training/africa/labels'
    t_len = 28
    n = 2

    pairs = build_pairs(root_s1, root_gt, root_labels, t_len)

    print(f"\nTotale tile: {len(pairs)}\n")

    for i, (gt, s1_list) in enumerate(pairs[: n]):
        print(f"[{i}] GT:", os.path.basename(gt))
        for t, s1 in enumerate(s1_list):
            print(f"    t={t}: {os.path.basename(s1)}")
        print("-" * 40)