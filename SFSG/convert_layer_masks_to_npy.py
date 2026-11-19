"""Convert ordered OCTA-6mm layer masks or native OCTA-500 annotations to NPY."""

import argparse
from pathlib import Path

import cv2
import numpy as np

HEIGHT, WIDTH = 640, 400
ENCODINGS = {
    "indices": (0, 1, 2, 3, 4, 5, 6),
    "grayscale": (0, 43, 85, 128, 170, 213, 255),
}


def mask_to_boundaries(mask, levels):
    """Return six first-row transition coordinates plus the two image edges."""
    if mask.shape != (HEIGHT, WIDTH):
        raise ValueError(f"Expected a {HEIGHT} x {WIDTH} single-channel mask, got {mask.shape}")

    ranks = np.searchsorted(levels, mask)
    if np.any(ranks >= len(levels)) or np.any(np.asarray(levels)[ranks] != mask):
        raise ValueError(f"Unexpected mask values {np.unique(mask).tolist()}; expected {list(levels)}")
    if np.any(np.diff(ranks, axis=0) < 0):
        raise ValueError("Mask classes must be ordered from top to bottom in each column")
    if np.any(~np.any((ranks > 0) & (ranks < 6), axis=0)):
        raise ValueError("At least one column contains no labeled retinal layer")

    boundaries = np.empty((8, WIDTH), dtype=np.int32)
    boundaries[0], boundaries[-1] = 0, HEIGHT
    for index in range(1, 7):
        boundaries[index] = np.count_nonzero(ranks < index, axis=0)
    return boundaries


def save_boundaries(path, boundaries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.save(output, boundaries, allow_pickle=False)


def convert_masks(input_root, output_root, levels):
    paths = sorted(input_root.rglob("*.png"))
    if not paths:
        raise ValueError(f"No PNG masks found in {input_root}")
    for path in paths:
        try:
            mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError("Could not decode PNG mask")
            boundaries = mask_to_boundaries(mask, levels)
            save_boundaries(output_root / path.relative_to(input_root).with_suffix(".npy"), boundaries)
        except (OSError, ValueError) as error:
            raise SystemExit(f"{path}: {error}") from error
    return len(paths)


def convert_mat(input_root, output_root):
    # Follows gen_layer_data.py's annotation layout and one-based slice filenames.
    from scipy.io import loadmat

    paths = sorted(input_root.rglob("*.mat"))
    if not paths:
        raise ValueError(f"No MAT annotations found in {input_root}")
    count = 0
    for path in paths:
        layers = loadmat(path, variable_names=["Layer"]).get("Layer")
        if (layers is None or layers.ndim != 3 or layers.shape[0] != 6
                or layers.shape[2] != WIDTH or not np.all(np.isfinite(layers))):
            raise ValueError(f"{path}: expected a finite Layer array of shape (6, slices, {WIDTH})")
        if np.any(layers < 0) or np.any(layers > HEIGHT) or np.any(np.diff(layers, axis=0) < 0):
            raise ValueError(f"{path}: invalid retinal boundary coordinates")
        volume = output_root / path.relative_to(input_root).with_suffix("")
        boundaries = np.empty((8, WIDTH), dtype=np.int32)
        boundaries[0], boundaries[-1] = 0, HEIGHT
        for index in range(layers.shape[1]):
            boundaries[1:7] = layers[:, index, :]
            save_boundaries(volume / f"{index + 1}.npy", boundaries)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--input-format", choices=("png", "mat"), default="png")
    parser.add_argument("--encoding", choices=ENCODINGS, default="indices",
                        help="Pixel encoding for PNG masks")
    args = parser.parse_args()

    source, target = args.input_root.resolve(), args.output_root.resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if target.is_relative_to(source):
        parser.error("Choose an output directory outside the input directory")
    if args.input_format == "mat":
        count = convert_mat(source, target)
    else:
        count = convert_masks(source, target, ENCODINGS[args.encoding])
    print(f"Saved {count} NPY files to {target}")


if __name__ == "__main__":
    main()
