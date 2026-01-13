import os
import re
import csv
import random
from pathlib import Path

# ----------------------------
# Edit these paths
# ----------------------------
PRE_IMG_FOLDER  = Path("/content/data/pre_disaster")
POST_IMG_FOLDER = Path("/content/data/post_disaster")
POST_JSON_FOLDER = Path("/content/labels/post_disaster")
MASK_FOLDER = Path("/content/train/masks")
SIXCH_FOLDER = Path("/content/train/inputs_6ch")

OUT_SPLIT_FOLDER = Path("/content/train/splits")
OUT_SPLIT_FOLDER.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Split ratios + seed
# ----------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED = 42

# ----------------------------
# Helpers
# ----------------------------
def scene_id_from_name(name: str) -> str:
    """
    Extract scene id from known suffix patterns.
    Example: hurricane-harvey_00000000_post_disaster.json -> hurricane-harvey_00000000
    """
    name = Path(name).name
    for suf in [
        "_pre_disaster.png", "_pre_disaster.jpg", "_pre_disaster.jpeg",
        "_post_disaster.png", "_post_disaster.jpg", "_post_disaster.jpeg",
        "_post_disaster.json",
        "_mask.png",
        "_6ch.npy",
    ]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return ""


def build_index(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    """Map scene_id -> file path for files in folder matching extensions."""
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.name.lower().endswith(exts):
            sid = scene_id_from_name(p.name)
            if sid:
                out[sid] = p
    return out


def main():
    # Index each required artifact by scene_id
    pre_idx   = build_index(PRE_IMG_FOLDER, (".png", ".jpg", ".jpeg"))
    post_idx  = build_index(POST_IMG_FOLDER, (".png", ".jpg", ".jpeg"))
    json_idx  = build_index(POST_JSON_FOLDER, (".json",))
    mask_idx  = build_index(MASK_FOLDER, (".png",))
    sixch_idx = build_index(SIXCH_FOLDER, (".npy",))

    # Choose what "complete" means for your pipeline:
    # If you already built 6ch + mask, use these as requirements (fastest).
    required_sets = [set(sixch_idx.keys()), set(mask_idx.keys())]

    # Optional: also require original images/jsons (uncomment if you want stricter)
    # required_sets += [set(pre_idx.keys()), set(post_idx.keys()), set(json_idx.keys())]

    common_ids = set.intersection(*required_sets)
    common_ids = sorted(common_ids)

    if not common_ids:
        print("No complete scenes found. Check folder paths and filename patterns.")
        return

    # Deterministic shuffle
    rng = random.Random(SEED)
    rng.shuffle(common_ids)

    n = len(common_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # remainder goes to test
    train_ids = common_ids[:n_train]
    val_ids = common_ids[n_train:n_train + n_val]
    test_ids = common_ids[n_train + n_val:]

    # Save ID lists
    (OUT_SPLIT_FOLDER / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (OUT_SPLIT_FOLDER / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    (OUT_SPLIT_FOLDER / "test_ids.txt").write_text("\n".join(test_ids) + "\n")

    # Save manifest CSV (handy later)
    manifest_path = OUT_SPLIT_FOLDER / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "split", "sixch_npy", "mask_png"])
        for sid in train_ids:
            w.writerow([sid, "train", str(sixch_idx[sid]), str(mask_idx[sid])])
        for sid in val_ids:
            w.writerow([sid, "val", str(sixch_idx[sid]), str(mask_idx[sid])])
        for sid in test_ids:
            w.writerow([sid, "test", str(sixch_idx[sid]), str(mask_idx[sid])])

    print(f"Total complete scenes: {n}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print(f"Saved splits to: {OUT_SPLIT_FOLDER}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
