"""Convert AROI 6 x 512 retinal-boundary NPY files to 1024 x 512 PNG masks."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


HEIGHT = 640
WIDTH = 400
EXPECTED_SHAPE = (6, WIDTH)
FILL_VALUES = (0, 64, 128, 192, 255)



def npy_to_mask(data):
    """Fill five regions between six ordered boundaries."""
    if data.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected NPY shape {EXPECTED_SHAPE}, got {data.shape}")
    if not np.issubdtype(data.dtype, np.number) or not np.all(np.isfinite(data)):
        raise ValueError("NPY must contain finite numeric coordinates")

    # Match the original script's int() conversion, then clip. The last
    # boundary is 1024 (exclusive image edge), so row 1023 is included.
    heights = np.clip(data, 0, HEIGHT).astype(np.int32)
    heights[0, :] = 0
    heights[5, :] = HEIGHT
    if np.any(np.diff(heights, axis=0) < 0):
        raise ValueError("Layer boundaries cross; smooth/project the NPY first")

    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for col in range(WIDTH):
        for region_idx, gray in enumerate(FILL_VALUES):
            top = heights[region_idx, col]
            bottom = heights[region_idx + 1, col]
            label[top:bottom, col] = gray
    return label


def convert_single_npy(input_path, output_path):
    data = np.load(input_path, allow_pickle=False)
    label = npy_to_mask(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded, image = cv2.imencode(".png", label)
    if not encoded:
        raise RuntimeError(f"PNG encoding failed: {output_path}")
    temporary = output_path.with_name("." + output_path.name + ".tmp")
    try:
        image.tofile(str(temporary))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def batch_convert(input_root, output_root, overwrite=False):
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("Output folder must be outside the input folder")

    files = sorted(input_root.rglob("*.npy"))
    success = skipped = failed = 0
    print(f"Found {len(files)} NPY files under {input_root}")
    for index, input_path in enumerate(files, 1):
        output_path = (output_root / input_path.relative_to(input_root)).with_suffix(".png")
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            convert_single_npy(input_path, output_path)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {input_path}: {exc}")
        if index % 500 == 0:
            print(f"Processed {index}/{len(files)} | saved {success} | skipped {skipped} | failed {failed}")

    print(f"Done | saved {success} | skipped {skipped} | failed {failed} | output {output_root}")
    if failed:
        raise RuntimeError(f"{failed} NPY files failed; see messages above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AROI 6 x 512 NPY layers to PNG masks")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute PNG files that already exist")
    args = parser.parse_args()
    batch_convert(args.input_root, args.output_root, args.overwrite)
