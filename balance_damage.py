import csv
import random
from pathlib import Path

TILES_MANIFEST = Path("/content/train/tiles_256/tiles_manifest.csv")
OUT_BALANCED_TRAIN = Path("/content/train/tiles_256/train_tiles_balanced.csv")

SEED = 42

# ----------------------------
# Balancing knobs (tune these)
# ----------------------------
# Keep only this fraction of empty tiles (no building pixels)
KEEP_EMPTY_RATIO = 0.05

# Oversample factors (how many times to repeat tiles in the final list)
# A tile can be in multiple categories; we’ll take the maximum factor that applies.
FACTOR_HAS_NO_DAMAGE = 1
FACTOR_HAS_MINOR     = 3
FACTOR_HAS_MAJOR     = 4
FACTOR_HAS_DESTROYED = 5


def category_factor(row: dict) -> int:
    """Return oversample factor based on the strongest damage class present in the tile."""
    # destroyed > major > minor > no-damage
    if row["has_destroyed"] == "1":
        return FACTOR_HAS_DESTROYED
    if row["has_major"] == "1":
        return FACTOR_HAS_MAJOR
    if row["has_minor"] == "1":
        return FACTOR_HAS_MINOR
    if row["has_no_damage"] == "1":
        return FACTOR_HAS_NO_DAMAGE
    return 1


def main():
    rng = random.Random(SEED)

    # Read all train tiles
    train_rows = []
    with open(TILES_MANIFEST, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "train":
                continue
            train_rows.append(row)

    if not train_rows:
        print("No train rows found. Check your tiles_manifest.csv and split names.")
        return

    # Separate empty vs non-empty
    empty = [r for r in train_rows if r["has_building"] == "0"]
    non_empty = [r for r in train_rows if r["has_building"] == "1"]

    # Subsample empty tiles
    kept_empty = [r for r in empty if rng.random() < KEEP_EMPTY_RATIO]

    # Oversample non-empty tiles by factor
    balanced = []
    for r in non_empty:
        k = category_factor(r)
        balanced.extend([r] * k)

    # Add kept empty tiles (usually not oversampled)
    balanced.extend(kept_empty)

    rng.shuffle(balanced)

    # Write balanced CSV
    OUT_BALANCED_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BALANCED_TRAIN, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sixch_tile_path", "mask_tile_path", "scene_id", "x", "y"])
        for r in balanced:
            writer.writerow([r["sixch_tile_path"], r["mask_tile_path"], r["scene_id"], r["x"], r["y"]])

    # Print quick stats
    def count_flag(rows, key):
        return sum(1 for r in rows if r[key] == "1")

    print("=== Original train tiles ===")
    print("Total:", len(train_rows))
    print("Empty:", len(empty))
    print("Non-empty:", len(non_empty))
    print("Has minor:", count_flag(non_empty, "has_minor"))
    print("Has major:", count_flag(non_empty, "has_major"))
    print("Has destroyed:", count_flag(non_empty, "has_destroyed"))

    print("\n=== Balanced train list ===")
    print("Total:", len(balanced))
    print("Kept empty:", len(kept_empty))
    # Approx presence stats in final list
    print("Tiles containing minor (in list):", sum(1 for r in balanced if r["has_minor"] == "1"))
    print("Tiles containing major (in list):", sum(1 for r in balanced if r["has_major"] == "1"))
    print("Tiles containing destroyed (in list):", sum(1 for r in balanced if r["has_destroyed"] == "1"))
    print("\nSaved:", OUT_BALANCED_TRAIN)


if __name__ == "__main__":
    main()
