import csv
from pathlib import Path

TILES_MANIFEST = Path("/content/train/tiles_256/tiles_manifest.csv")
OUT_DIR = Path("/content/train/tiles_256")

def write_split_csv(split_name: str, out_path: Path):
    with open(TILES_MANIFEST, "r", newline="") as f_in, open(out_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["sixch_tile_path", "mask_tile_path", "scene_id", "x", "y"])
        n = 0
        for r in reader:
            if r["split"] != split_name:
                continue
            writer.writerow([r["sixch_tile_path"], r["mask_tile_path"], r["scene_id"], r["x"], r["y"]])
            n += 1
    print(f"Wrote {n} rows -> {out_path}")

if __name__ == "__main__":
    write_split_csv("val",  OUT_DIR / "val_tiles.csv")
    write_split_csv("test", OUT_DIR / "test_tiles.csv")
