"""Generate spatially-blocked cross-validation folds for the bubble U-Net.

WHY
    The frozen hold-out in data/processed/metadata/valRegions.csv is two
    rectangles yielding 21 validation patches. At stride 128 on 256px patches
    neighbours overlap by 50%, so those 21 patches resample roughly 8
    non-overlapping tiles of sky. Stage-to-stage differences of ~0.01 Dice are
    not measurable against that, and extra seeds cannot fix it - seeds address
    model variance, not the size of the measuring instrument.

    This script partitions the map into K contiguous blocks. Each block serves
    as the hold-out once, giving K estimates instead of one, so a curriculum
    gain can be quoted as a mean with a spread.

LEAKAGE
    Blocks are contiguous and the split rule is inherited unchanged from
    compositePatches.assign_split: a patch counts as val only if it lies fully
    inside the held-out block, and any patch straddling the boundary is
    excluded from both sides. A training patch therefore never shares a pixel
    with a validation patch. The excluded band is the price of that guarantee
    and is reported per fold below.

Writes  data/processed/metadata/folds/fold{k}/valRegions.csv

Usage:
    python scripts/makeFolds.py                    # K=3 vertical strips
    python scripts/makeFolds.py --k 4 --rows 2     # 2x2 grid blocks
    python scripts/makeFolds.py --dry-run          # report only, write nothing

Then, per (stage, fold):
    python scripts/compositePatches.py \
        --mask data/processed/masks/<stage>.npy \
        --val-regions data/processed/metadata/folds/fold0/valRegions.csv \
        --out-dir data/processed/patches_composite/<stage>/fold0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

MASK_PATH = REPO / "data/processed/masks/ngc628NormMask.npy"
OUTPUT_DIR = REPO / "data/processed/metadata/folds"

# must match compositePatches.py or the diagnostics below are fiction
PATCH_SIZE = 256
STRIDE = 128
LOW_THRESHOLD = 0.001
HIGH_THRESHOLD = 0.05


def make_blocks(height, width, n_rows, n_cols):
    """Contiguous blocks covering the map; trailing block absorbs the remainder."""
    bh, bw = height // n_rows, width // n_cols
    blocks = []
    for r in range(n_rows):
        for c in range(n_cols):
            blocks.append((
                r * bh,
                c * bw,
                bh if r < n_rows - 1 else height - r * bh,
                bw if c < n_cols - 1 else width - c * bw,
            ))
    return blocks


def assign_split(y, x, regions, patch_size=PATCH_SIZE):
    """Identical semantics to compositePatches.assign_split."""
    for (y0, x0, h, w) in regions:
        y1, x1 = y0 + h, x0 + w
        if y >= y0 and y + patch_size <= y1 and x >= x0 and x + patch_size <= x1:
            return "val"
        if not (y + patch_size <= y0 or y >= y1 or
                x + patch_size <= x0 or x >= x1):
            return "excluded"
    return "train"


def patch_grid(mask):
    H, W = mask.shape
    return [
        (y, x, float(mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE].mean()))
        for y in range(0, H - PATCH_SIZE + 1, STRIDE)
        for x in range(0, W - PATCH_SIZE + 1, STRIDE)
    ]


def categorise(ratio):
    if ratio >= HIGH_THRESHOLD:
        return "positive"
    if ratio >= LOW_THRESHOLD:
        return "weak"
    return "negative"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=3, help="number of folds")
    parser.add_argument("--rows", type=int, default=1,
                        help="block rows; K/rows must be a whole number")
    parser.add_argument("--mask", type=Path, default=MASK_PATH,
                        help="any stage mask - only the grid shape and the "
                             "positive/negative diagnostics depend on it")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.k % args.rows:
        raise SystemExit(f"--k {args.k} is not divisible by --rows {args.rows}")
    n_cols = args.k // args.rows

    if not args.mask.exists():
        raise SystemExit(f"mask not found: {args.mask}")
    mask = np.load(args.mask)
    H, W = mask.shape
    grid = patch_grid(mask)

    blocks = make_blocks(H, W, args.rows, n_cols)

    print(f"map {H}x{W}   patch {PATCH_SIZE}  stride {STRIDE}")
    print(f"{len(grid)} grid positions   layout {args.rows}x{n_cols} = {args.k} folds\n")

    header = (f"{'fold':<6}{'val':>6}{'val_pos':>9}{'train':>8}"
              f"{'excl':>7}   block (y, x, h, w)")
    print(header)
    print("-" * (len(header) + 4))

    rows, total_val, total_excl, weakest = [], 0, 0, None
    for k, block in enumerate(blocks):
        regions = [block]
        splits = [assign_split(y, x, regions) for y, x, _ in grid]
        n_val = sum(s == "val" for s in splits)
        n_pos = sum(s == "val" and categorise(r) == "positive"
                    for s, (_, _, r) in zip(splits, grid))
        n_train = sum(s == "train" for s in splits)
        n_excl = sum(s == "excluded" for s in splits)

        total_val += n_val
        total_excl += n_excl
        weakest = n_pos if weakest is None else min(weakest, n_pos)

        y0, x0, h, w = block
        print(f"{k:<6}{n_val:>6}{n_pos:>9}{n_train:>8}{n_excl:>7}   "
              f"({y0}, {x0}, {h}, {w})")
        rows.append((k, block, n_val, n_pos, n_train, n_excl))

    print(f"\ntotal val {total_val} patches across folds "
          f"(current fixed hold-out: 21)")
    print(f"total excluded at boundaries: {total_excl}")
    print(f"weakest fold carries {weakest} positive val patches")

    if weakest is not None and weakest < 10:
        print("\nWARNING: a fold with <10 positive validation patches will "
              "produce a noisy Dice estimate and can drag the fold mean "
              "around. Prefer fewer, larger blocks.")

    if args.dry_run:
        print("\ndry run - no fold files written")
        return

    for k, block, *_ in rows:
        fold_dir = args.out_dir / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        y0, x0, h, w = block
        pd.DataFrame([{"y_start": y0, "x_start": x0, "height": h, "width": w}]) \
          .to_csv(fold_dir / "valRegions.csv", index=False)

    print(f"\nwrote {len(rows)} fold definitions under {args.out_dir}")
    print("each fold0/valRegions.csv is a drop-in replacement for "
          "data/processed/metadata/valRegions.csv")


if __name__ == "__main__":
    main()
