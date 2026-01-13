import os
import numpy as np
from PIL import Image

# ----------------------------
# Set your folders
# ----------------------------
PRE_IMG_FOLDER  = "/content/data/pre_disaster"
POST_IMG_FOLDER = "/content/data/post_disaster"
OUTPUT_6CH_FOLDER = "/content/train/inputs_6ch"

# If you want float in [0,1] set True; otherwise keep uint8 0..255
NORMALIZE_TO_0_1 = True


def load_rgb(path: str) -> np.ndarray:
    """Loads an image as RGB numpy array (H,W,3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def make_6ch(pre_rgb: np.ndarray, post_rgb: np.ndarray) -> np.ndarray:
    """Stack pre and post into (H,W,6)."""
    if pre_rgb.shape != post_rgb.shape:
        raise ValueError(f"Size mismatch: pre {pre_rgb.shape} vs post {post_rgb.shape}")
    x = np.concatenate([pre_rgb, post_rgb], axis=-1)  # (H,W,6)
    if NORMALIZE_TO_0_1:
        x = x.astype(np.float32) / 255.0
    return x


def match_post_name(pre_name: str) -> str:
    """
    Converts a pre filename to its post filename.
    Example: hurricane-harvey_00000000_pre_disaster.png -> ..._post_disaster.png
    """
    return pre_name.replace("_pre_disaster", "_post_disaster")


def main():
    os.makedirs(OUTPUT_6CH_FOLDER, exist_ok=True)

    pre_files = sorted([
        f for f in os.listdir(PRE_IMG_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not pre_files:
        print("No pre images found in:", PRE_IMG_FOLDER)
        return

    ok, skipped, failed = 0, 0, 0

    for pre_name in pre_files:
        post_name = match_post_name(pre_name)

        pre_path = os.path.join(PRE_IMG_FOLDER, pre_name)
        post_path = os.path.join(POST_IMG_FOLDER, post_name)

        if not os.path.exists(post_path):
            skipped += 1
            print(f"[SKIP] No matching post image for: {pre_name}")
            continue

        try:
            pre_rgb = load_rgb(pre_path)
            post_rgb = load_rgb(post_path)
            x6 = make_6ch(pre_rgb, post_rgb)

            # Output name: replace pre_disaster with 6ch
            out_name = pre_name.replace("_pre_disaster", "_6ch")
            out_name = os.path.splitext(out_name)[0] + ".npy"
            out_path = os.path.join(OUTPUT_6CH_FOLDER, out_name)

            np.save(out_path, x6)
            ok += 1

            if ok <= 5:
                print(f"[OK] {pre_name} + {post_name} -> {out_name} | shape={x6.shape} dtype={x6.dtype}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {pre_name}: {e}")

    print(f"\nDone. Saved {ok} files to: {OUTPUT_6CH_FOLDER}")
    print(f"Skipped (no post match): {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()


# import numpy as np

# x = np.load("path/to/example_6ch.npy")

# print("Shape:", x.shape)        # should be (1024, 1024, 6)
# print("Dtype:", x.dtype)        # float32 or uint8
# print("Min / Max:", x.min(), x.max())