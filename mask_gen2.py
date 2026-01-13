import os
import json
import re
import numpy as np
from PIL import Image, ImageDraw

# ----------------------------
# Paths (yours)
# ----------------------------
JSON_FOLDER = "/content/labels/post_disaster"
OUTPUT_MASK_FOLDER = "/content/train/masks"

# ----------------------------
# Label mapping (5 classes)
# ----------------------------
DAMAGE_TO_ID = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
# background = 0


def parse_wkt_polygon(wkt: str):
    """
    Supports:
      POLYGON ((x y, x y, ...))
      POLYGON ((outer...), (hole...))  (holes optional)
    Returns: list of rings, each ring is list[(x,y)]
    """
    wkt = wkt.strip()
    if not wkt.startswith("POLYGON"):
        raise ValueError(f"Unsupported geometry (expected POLYGON): {wkt[:30]}...")

    inner = wkt[wkt.find("("):].strip()   # "((" ... "))"
    inner = inner[2:-2].strip()           # remove outer "((" and "))"
    ring_strs = re.split(r"\)\s*,\s*\(", inner)

    rings = []
    for ring_str in ring_strs:
        pts = []
        for pair in ring_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            x_str, y_str = pair.split()
            pts.append((float(x_str), float(y_str)))
        if len(pts) >= 3:
            rings.append(pts)

    if not rings:
        raise ValueError("Failed to parse polygon rings.")
    return rings


def clamp_points(points, width, height):
    out = []
    for x, y in points:
        x = 0.0 if x < 0 else (width - 1.0 if x > width - 1 else x)
        y = 0.0 if y < 0 else (height - 1.0 if y > height - 1 else y)
        out.append((x, y))
    return out


def build_mask_from_post_json(post_json_path: str):
    with open(post_json_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    width = int(meta.get("width", meta.get("original_width", 1024)))
    height = int(meta.get("height", meta.get("original_height", 1024)))

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    feats = data.get("features", {}).get("xy", [])
    buildings = []

    for feat in feats:
        props = feat.get("properties", {})
        if props.get("feature_type") != "building":
            continue

        subtype = (props.get("subtype") or "").strip()
        if subtype not in DAMAGE_TO_ID:
            # skip unknown/un-classified labels
            continue

        wkt = feat.get("wkt", "")
        if not wkt:
            continue

        buildings.append((DAMAGE_TO_ID[subtype], wkt))

    # overlap rule: max damage wins -> draw low first, high last
    buildings.sort(key=lambda t: t[0])

    for label_id, wkt in buildings:
        rings = parse_wkt_polygon(wkt)

        outer = clamp_points(rings[0], width, height)
        draw.polygon(outer, fill=int(label_id))

        # holes (rare): paint back to background
        for hole_ring in rings[1:]:
            hole = clamp_points(hole_ring, width, height)
            draw.polygon(hole, fill=0)

    return np.array(mask_img, dtype=np.uint8), meta


def mask_filename(meta: dict, json_path: str):
    """
    Prefer naming from metadata.img_name but convert post_disaster.png -> mask.png.
    Fallback to json filename.
    """
    img_name = meta.get("img_name")
    if isinstance(img_name, str) and img_name:
        base = os.path.basename(img_name)
        base = base.replace("_post_disaster.png", "_mask.png")
        base = base.replace("_post_disaster.jpg", "_mask.png")
        base = base.replace(".png", "_mask.png") if base.endswith("_post_disaster") else base
        return base

    base = os.path.splitext(os.path.basename(json_path))[0]
    return f"{base}_mask.png"


def main():
    os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(JSON_FOLDER)
        if f.lower().endswith(".json")
    )

    if not json_files:
        print("No .json files found in:", JSON_FOLDER)
        return

    ok, failed = 0, 0

    for jf in json_files:
        in_path = os.path.join(JSON_FOLDER, jf)
        try:
            mask, meta = build_mask_from_post_json(in_path)
            out_name = mask_filename(meta, in_path)
            out_path = os.path.join(OUTPUT_MASK_FOLDER, out_name)

            Image.fromarray(mask).save(out_path)

            # quick sanity: values should be subset of 0..4
            vals = np.unique(mask)
            if np.any(vals > 4):
                print(f"[WARN] {jf}: unexpected values {vals} -> check parsing/rasterization")

            ok += 1
            if ok <= 5:
                print(f"[OK] {jf} -> {out_name} | shape={mask.shape} | unique={vals}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {jf}: {e}")

    print(f"\nDone. Saved {ok} masks to: {OUTPUT_MASK_FOLDER}")
    if failed:
        print(f"Failed: {failed} file(s). Check logs above.")

if __name__ == "__main__":
    main()
