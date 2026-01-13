import os
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------
# Paths (edit these)
# ----------------------------
SPLIT_MANIFEST = Path("/content/train/splits/manifest.csv")

OUT_TILE_ROOT = Path("/content/train/tiles_256")
# Output structure:
# tiles_256/
#   train/inputs/*.npy
#   train/masks/*.png
#   val/inputs/*.npy
#   val/masks/*.png
#   test/inputs/*.npy
#   test/masks/*.png

# ----------------------------
# Tiling config
# ----------------------------
TILE_SIZE = 256
STRIDE = 128          # set 128 for 50% overlap
PAD_TO_FIT = False    # if True, pad images so edges fit a full tile

# Keep all tiles that contain any building pixels (mask > 0).
# For empty tiles (mask all zeros), keep only this fraction:
KEEP_EMPTY_RATIO = 0.10

SEED = 42


def pad_to_multiple(arr, mult, is_mask=False):
    """Pad H/W to a multiple of mult. arr can be (H,W) mask or (H,W,C) image."""
    h, w = arr.shape[:2]
    new_h = ((h + mult - 1) // mult) * mult
    new_w = ((w + mult - 1) // mult) * mult
    pad_h = new_h - h
    pad_w = new_w - w

    if pad_h == 0 and pad_w == 0:
        return arr

    if is_mask:
        # pad with background (0)
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    else:
        # pad image with zeros
        if arr.ndim == 3:
            return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


def ensure_dirs(split_name: str):
    (OUT_TILE_ROOT / split_name / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT_TILE_ROOT / split_name / "masks").mkdir(parents=True, exist_ok=True)


def tile_scene(scene_id: str, split_name: str, sixch_path: Path, mask_path: Path, rng: random.Random, writer):
    x6 = np.load(sixch_path)  # (H,W,6)
    mask = np.array(Image.open(mask_path), dtype=np.uint8)  # (H,W)

    if x6.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Size mismatch for {scene_id}: x6 {x6.shape} vs mask {mask.shape}")

    if PAD_TO_FIT:
        x6 = pad_to_multiple(x6, TILE_SIZE, is_mask=False)
        mask = pad_to_multiple(mask, TILE_SIZE, is_mask=True)

    H, W = mask.shape
    split_in_dir = OUT_TILE_ROOT / split_name / "inputs"
    split_m_dir = OUT_TILE_ROOT / split_name / "masks"

    tile_count = 0
    kept_count = 0

    # iterate tiles
    for y in range(0, H - TILE_SIZE + 1, STRIDE):
        for x in range(0, W - TILE_SIZE + 1, STRIDE):
            tile_count += 1
            x_tile = x6[y:y+TILE_SIZE, x:x+TILE_SIZE, :]
            m_tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]

            has_building = bool((m_tile > 0).any())
            is_empty = not has_building

            # filtering rule
            if is_empty and rng.random() > KEEP_EMPTY_RATIO:
                continue

            kept_count += 1

            tile_name = f"{scene_id}_x{x}_y{y}"
            x_out = split_in_dir / f"{tile_name}.npy"
            m_out = split_m_dir / f"{tile_name}.png"

            np.save(x_out, x_tile)
            Image.fromarray(m_tile).save(m_out)

            # write tile manifest row
            writer.writerow([
                split_name, scene_id, x, y,
                str(x_out), str(m_out),
                int(has_building),
                int((m_tile == 1).any()),  # has no-damage
                int((m_tile == 2).any()),  # has minor
                int((m_tile == 3).any()),  # has major
                int((m_tile == 4).any()),  # has destroyed
            ])

    return tile_count, kept_count


def main():
    rng = random.Random(SEED)

    ensure_dirs("train")
    ensure_dirs("val")
    ensure_dirs("test")

    out_manifest = OUT_TILE_ROOT / "tiles_manifest.csv"
    with open(SPLIT_MANIFEST, "r", newline="") as f_in, open(out_manifest, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)

        writer.writerow([
            "split", "scene_id", "x", "y",
            "sixch_tile_path", "mask_tile_path",
            "has_building", "has_no_damage", "has_minor", "has_major", "has_destroyed"
        ])

        totals = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}  # [all_tiles, kept_tiles]

        for row in reader:
            scene_id = row["scene_id"]
            split_name = row["split"].strip().lower()
            sixch_path = Path(row["sixch_npy"])
            mask_path = Path(row["mask_png"])

            if split_name not in totals:
                print(f"[SKIP] Unknown split '{split_name}' for {scene_id}")
                continue

            try:
                all_tiles, kept_tiles = tile_scene(scene_id, split_name, sixch_path, mask_path, rng, writer)
                totals[split_name][0] += all_tiles
                totals[split_name][1] += kept_tiles
                print(f"[OK] {scene_id} ({split_name}) -> tiles kept {kept_tiles}/{all_tiles}")
            except Exception as e:
                print(f"[FAIL] {scene_id}: {e}")

    print("\nDone.")
    for sp, (all_t, kept_t) in totals.items():
        print(f"{sp}: kept {kept_t}/{all_t} tiles")
    print("Tile manifest saved to:", out_manifest)
    print("Tiles saved under:", OUT_TILE_ROOT)


if __name__ == "__main__":
    main()
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# x = np.load("/Users/chimdiaanyiam/Desktop/school/dissertation/train/tiles_256/train/inputs/hurricane-harvey_00000000_x0_y0.npy")
# m = np.array(Image.open("/Users/chimdiaanyiam/Desktop/school/dissertation/train/tiles_256/train/masks/hurricane-harvey_00000000_x0_y0.png"))

# plt.imshow(x[:, :, 3:6])      # post RGB
# plt.imshow(m, alpha=0.4, cmap="jet")
# plt.axis("off")
# plt.show()