
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from scipy import ndimage

import numpy as np

import tensorflow as tf
from astropy.io import fits
from astropy.wcs import WCS

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


REPO = Path(__file__).resolve().parents[1]

THRESHOLD = 0.5      # prediction binarization, same as compositeEval.py
MIN_BLOB_PX = 20     # ignore predicted blobs smaller than this
WATERSHED_MIN_DISTANCE = 2
MIN_AXIS_RATIO = 0.412
LABEL_TOP_N = 30     # number the N most probable candidates in the overlay
STRIDE = 128         # tile stride, same overlap as training patches
DISTANCE_PC = 9.77e6 # NGC 628; override with --distance-pc for other targets


def first_image_hdu(hdul):
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return hdu
    raise ValueError("no 2D image HDU found")


def normalize_single_band(data):
    data_clean = np.nan_to_num(data, nan=0.0)
    p1, p99 = np.percentile(data_clean, (1, 99))
    return ((np.clip(data_clean, p1, p99) - p1) / (p99 - p1)).astype(np.float32)


def tile_positions(length, patch, stride):
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def predict_probability_map(image, model, patch, stride, batch_size):
    H, W = image.shape[:2]
    tiles = [(y, x)
             for y in tile_positions(H, patch, stride)
             for x in tile_positions(W, patch, stride)]

    prob_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for start in range(0, len(tiles), batch_size):
        batch = tiles[start:start + batch_size]
        Xb = np.stack([image[y:y + patch, x:x + patch] for y, x in batch])
        Pb = np.squeeze(model.predict(Xb, verbose=0), axis=-1)
        for (y, x), p in zip(batch, Pb):
            prob_sum[y:y + patch, x:x + patch] += p
            count[y:y + patch, x:x + patch] += 1.0
        done = min(start + batch_size, len(tiles))
        print(f"\r    tiles {done}/{len(tiles)}", end="", flush=True)
    print()

    return prob_sum / np.maximum(count, 1.0)


def label_blobs(mask, watershed_min_distance=None):
    if watershed_min_distance is None:
        return ndimage.label(mask)

    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    distance = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(distance, labels=mask.astype(int),
                            min_distance=watershed_min_distance,
                            exclude_border=False)
    if len(coords) == 0:
        return ndimage.label(mask)
    seeds = np.zeros(mask.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = ndimage.label(seeds)
    labels = watershed(-distance, markers, mask=mask)
    return labels, int(labels.max())


def tuned_threshold(model_path):
    prefix = model_path.stem.replace("bubble_unet", "").replace("_best.weights", "")
    summary = REPO / "outputs" / "history" / f"{prefix}threshold_summary.json"
    if not summary.exists():
        return None
    try:
        return float(json.loads(summary.read_text())["best_threshold"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def combine_maps(probs, thresholds, mode, mean_threshold):
    mean_map = np.stack(probs).mean(axis=0)
    if mode == "mean":
        return (mean_map > mean_threshold, mean_map, None,
                f"mean > {mean_threshold:g}")

    # Each model votes at its OWN calibrated threshold, so a fold that runs hot
    # and one that runs cold contribute comparable votes.
    votes = np.zeros(mean_map.shape, dtype=np.int16)
    for p, t in zip(probs, thresholds):
        votes += (p > t).astype(np.int16)
    n = len(probs)
    agreement = votes / n
    if mode == "unanimous":
        return votes == n, mean_map, agreement, f"all {n} models agree"
    return (votes * 2 > n, mean_map, agreement,
            f"more than {n // 2} of {n} models agree")


def extract_candidates(binary, prob, min_blob_px, watershed_min_distance=None,
                       min_axis_ratio=0.0, agreement=None,
                       wcs=None, arcsec_per_pix=None, pc_per_pixel=None):
    from skimage.measure import regionprops

    labels, n = label_blobs(binary, watershed_min_distance)
    mask = np.zeros(prob.shape, dtype=np.uint8)
    kept = np.zeros(prob.shape, dtype=np.int32)
    rows = []
    for prop in regionprops(labels, intensity_image=prob):
        area = int(prop.area)
        if area < min_blob_px:
            continue
        # A spiral-arm segment is elongated far beyond anything in the training
        # catalogue; see MIN_AXIS_RATIO for where the bound comes from.
        major = float(prop.axis_major_length)
        axis_ratio = float(prop.axis_minor_length) / major if major > 0 else 0.0
        if axis_ratio < min_axis_ratio:
            continue

        y_pix, x_pix = prop.centroid
        equiv_radius_px = float(np.sqrt(area / np.pi))
        candidate_id = len(rows) + 1
        y0, x0, y1, x1 = prop.bbox
        mask[y0:y1, x0:x1][prop.image] = 1
        kept[y0:y1, x0:x1][prop.image] = candidate_id
        row = {
            "candidate_id": candidate_id,
            "x_pix": round(float(x_pix), 2),
            "y_pix": round(float(y_pix), 2),
            "area_px": area,
            "equiv_radius_px": round(equiv_radius_px, 2),
            "axis_ratio": round(axis_ratio, 3),
            "mean_prob": round(float(prop.intensity_mean), 4),
            "max_prob": round(float(prop.intensity_max), 4),
            "y_min": y0, "y_max": y1,
            "x_min": x0, "x_max": x1,
        }
        if agreement is not None:
            row["mean_agreement"] = round(
                float(agreement[y0:y1, x0:x1][prop.image].mean()), 3)
        if wcs is not None:
            ra, dec = wcs.all_pix2world(x_pix, y_pix, 0)
            row["ra_deg"] = round(float(ra), 6)
            row["dec_deg"] = round(float(dec), 6)
        if arcsec_per_pix is not None:
            row["equiv_radius_arcsec"] = round(equiv_radius_px * arcsec_per_pix, 3)
        if pc_per_pixel is not None:
            row["equiv_radius_pc"] = round(equiv_radius_px * pc_per_pixel, 1)
        rows.append(row)

    return mask, kept, pd.DataFrame(rows)


def write_region_file(path, candidates, arcsec_per_pix):
    lines = ["# Region file format: DS9 version 4.1",
             "global color=cyan width=1 font=\"helvetica 8 normal\""]
    sky = "ra_deg" in candidates.columns and arcsec_per_pix is not None
    lines.append("fk5" if sky else "image")
    for _, c in candidates.iterrows():
        if sky:
            lines.append(f'circle({c.ra_deg:.6f}d,{c.dec_deg:.6f}d,'
                         f'{c.equiv_radius_px * arcsec_per_pix:.2f}")'
                         f' # text={{{int(c.candidate_id)}}}')
        else:  # DS9 image coordinates are 1-based
            lines.append(f"circle({c.x_pix + 1:.1f},{c.y_pix + 1:.1f},"
                         f"{c.equiv_radius_px:.1f})"
                         f" # text={{{int(c.candidate_id)}}}")
    path.write_text("\n".join(lines) + "\n")


def sha256_of(path, chunk=2 ** 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, required=True,
                        help="raw FITS mosaic or preprocessed .npy in [0,1]")
    parser.add_argument("--model", type=Path, required=True, nargs="+",
                        help="trained .keras model")
    parser.add_argument("--output", type=Path, required=True,
                        help="output directory (created if missing)")
    parser.add_argument("--wcs-from", type=Path, default=None, metavar="FITS",
                        help="FITS file supplying the WCS when --image is a .npy")
    parser.add_argument("--threshold", type=float, default=None,
                        help="binarization threshold. Default: the value tuned "
                             "on validation and recorded in "
                             "outputs/history/<model>threshold_summary.json, "
                             f"falling back to {THRESHOLD} when there is no "
                             "such record. Only used by --combine mean; the "
                             "vote modes always cut each model at its own "
                             "tuned threshold")
    parser.add_argument("--combine", choices=("mean", "majority", "unanimous"),
                        default="majority",
                        help="how to reduce several models to one detection: "
                             "average the probability maps, or count votes cast "
                             "at each model's own tuned threshold. Ignored for "
                             "a single model (default: majority)")
    parser.add_argument("--min-blob-px", type=int, default=MIN_BLOB_PX)
    parser.add_argument("--watershed-min-distance", type=int,
                        default=WATERSHED_MIN_DISTANCE, metavar="PX",
                        help="minimum spacing between watershed seeds; "
                             f"default {WATERSHED_MIN_DISTANCE} px, derived as "
                             "the radius of the smallest bubble in the training "
                             "catalogue (6 pc at NGC 628's 5.25 pc/px)")
    parser.add_argument("--min-axis-ratio", type=float, default=MIN_AXIS_RATIO,
                        metavar="R",
                        help="discard blobs flatter than this minor/major axis "
                             f"ratio; default {MIN_AXIS_RATIO}, the smallest "
                             "ratio in the training catalogue. 0 disables")
    parser.add_argument("--label-top", type=int, default=LABEL_TOP_N, metavar="N",
                        help=f"number the N most probable candidates in the "
                             f"overlay (default {LABEL_TOP_N}); 0 labels none")
    parser.add_argument("--no-watershed", action="store_true",
                        help="label connected components without splitting "
                             "merged blobs (the pre-2026-08 behaviour)")
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--distance-pc", type=float, default=DISTANCE_PC,
                        help="distance to the target for radii in parsec")
    args = parser.parse_args()

    t0 = time.time()


    models = []
    for path in args.model:
        print(f"loading model {path} ...")
        m = tf.keras.models.load_model(path, compile=False)
        t = tuned_threshold(path)
        if t is None:
            t = args.threshold if args.threshold is not None else THRESHOLD
            print(f"    no threshold_summary.json - using {t}")
        else:
            print(f"    tuned threshold {t:.3f}")
        models.append({"path": path, "model": m, "threshold": t})

    shapes = {tuple(d["model"].input_shape[1:]) for d in models}
    if len(shapes) > 1:
        sys.exit(f"models disagree on input shape: {sorted(shapes)}")
    patch = models[0]["model"].input_shape[1]
    model_channels = models[0]["model"].input_shape[-1]

    print(f"loading image {args.image} ...")
    header = wcs = footprint = None
    if args.image.suffix.lower() in (".fits", ".fit"):
        with fits.open(args.image) as hdul:
            hdu = first_image_hdu(hdul)
            raw = hdu.data.astype(np.float64)
            header = hdu.header
        wcs = WCS(header)
        footprint = np.isfinite(raw)
        image = normalize_single_band(raw)[..., np.newaxis]
        wcs_source = str(args.image)
    else:
        image = np.load(args.image).astype(np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if image.min() < 0 or image.max() > 1.5:
            print(f"    WARNING: values outside [0,1] "
                  f"(min={image.min():.3g}, max={image.max():.3g}) - "
                  "is this really a preprocessed image?")
        wcs_source = None
        if args.wcs_from is not None:
            with fits.open(args.wcs_from) as hdul:
                header = first_image_hdu(hdul).header
            wcs = WCS(header)
            wcs_source = str(args.wcs_from)
        else:
            print("    no --wcs-from: candidate_regions will have "
                  "pixel coordinates only")

    H, W, channels = image.shape
    print(f"    grid ({H}, {W}), {channels} channel(s)")
    if channels != model_channels:
        sys.exit(f"model expects {model_channels} channel(s), image has "
                 f"{channels} - single-band model needs the FITS/ngc628Norm, "
                 "composite model needs ngc628Composite.npy")
    if H < patch or W < patch:
        sys.exit(f"image smaller than the model input ({patch}x{patch})")
    if wcs is not None and wcs.pixel_shape is not None:
        if (wcs.pixel_shape[1], wcs.pixel_shape[0]) != (H, W):
            sys.exit(f"--wcs-from grid {wcs.pixel_shape} does not match "
                     f"image grid ({W}, {H})")

    arcsec_per_pix = pc_per_pixel = None
    if header is not None and "CDELT1" in header:
        arcsec_per_pix = abs(header["CDELT1"]) * 3600
        pc_per_pixel = arcsec_per_pix * args.distance_pc / 206265

    probs = []
    for d in models:
        print(f"predicting with {d['path'].name} "
              f"(patch {patch}, stride {args.stride}) ...")
        p = predict_probability_map(image, d["model"], patch, args.stride,
                                    args.batch_size)
        if footprint is not None:
            p[~footprint] = 0.0  # no claims outside the mosaic coverage
        probs.append(p)

    mean_threshold = args.threshold
    if mean_threshold is None:
        mean_threshold = (models[0]["threshold"] if len(models) == 1
                          else float(np.mean([d["threshold"] for d in models])))
    if len(models) == 1:
        combine_mode = "mean"     # a single model has nothing to vote against
    else:
        combine_mode = args.combine
    binary, prob, agreement, combine_desc = combine_maps(
        probs, [d["threshold"] for d in models], combine_mode, mean_threshold)
    print(f"combining {len(models)} model(s): {combine_desc}")

    ws_distance = None if args.no_watershed else args.watershed_min_distance
    print(f"labelling blobs ("
          + (f"watershed, seed spacing {ws_distance} px"
             if ws_distance else "connected components, no watershed") + ") ...")
    mask, kept_labels, candidates = extract_candidates(
        binary, prob, args.min_blob_px, ws_distance,
        min_axis_ratio=args.min_axis_ratio, agreement=agreement,
        wcs=wcs, arcsec_per_pix=arcsec_per_pix, pc_per_pixel=pc_per_pixel,
    )
    print(f"    {len(candidates)} candidate region(s) "
          f"({combine_desc}, min {args.min_blob_px} px, "
          f"axis ratio >= {args.min_axis_ratio})")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "probability_map.npy", prob)
    np.save(out / "binary_mask.npy", mask)

    if agreement is not None:
        np.save(out / "agreement_map.npy", agreement)

    hdr = wcs.to_header() if wcs is not None else fits.Header()
    hdr["BUNIT"] = "probability"
    hdr.add_history("where-is-my-bubble models="
                    + ",".join(p.name for p in args.model))
    hdr.add_history(f"combine={combine_mode} ({combine_desc})")
    fits.PrimaryHDU(prob, header=hdr).writeto(out / "probability_map.fits",
                                              overwrite=True)

    candidates.to_csv(out / "candidate_regions.csv", index=False)
    write_region_file(out / "candidate_regions.reg", candidates, arcsec_per_pix)

    matplotlib.use("Agg")

    from skimage.segmentation import find_boundaries

    fig, ax = plt.subplots(figsize=(12, 12))
    if channels == 3:
        ax.imshow(image, origin="lower")
    else:
        flat = np.squeeze(image)
        finite = flat[flat > 0]
        vmin, vmax = (np.percentile(finite, (1, 99.5)) if finite.size
                      else (0.0, 1.0))
        ax.imshow(flat, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    edges = find_boundaries(kept_labels, mode="inner")
    overlay = np.zeros((*kept_labels.shape, 4), dtype=float)
    overlay[edges] = (1.0, 0.15, 0.15, 1.0)
    ax.imshow(overlay, origin="lower", interpolation="nearest")

    if args.label_top > 0 and len(candidates):
        top = candidates.nlargest(args.label_top, "mean_prob")
        for _, c in top.iterrows():
            ax.annotate(str(int(c.candidate_id)),
                        (c.x_pix, c.y_pix + c.equiv_radius_px + 6),
                        color="yellow", fontsize=6, ha="center",
                        path_effects=[pe.withStroke(linewidth=1.4,
                                                    foreground="black")])
        ax.scatter(top.x_pix, top.y_pix, s=6, facecolors="none",
                   edgecolors="yellow", linewidths=0.5)

    subtitle = (combine_desc + " | "
                + (f"watershed d={ws_distance}" if ws_distance else "no watershed")
                + f" | axis ratio >= {args.min_axis_ratio}")
    names = ", ".join(p.name for p in args.model)
    ax.set_title(f"{args.image.name} | {len(models)} model(s): {names}\n"
                 f"{len(candidates)} candidates | {subtitle}", fontsize=8)
    ax.axis("off")
    plt.savefig(out / "prediction_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None

    metadata = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t0, 1),
        "image": {"path": str(args.image), "grid": [H, W],
                  "channels": channels, "wcs_source": wcs_source},
        "models": [{"path": str(d["path"]), "sha256": sha256_of(d["path"]),
                    "threshold": d["threshold"],
                    "threshold_source": ("tuned"
                                         if tuned_threshold(d["path"]) is not None
                                         else "fallback")}
                   for d in models],
        "input_shape": list(models[0]["model"].input_shape[1:]),
        "parameters": {"combine": combine_mode,
                       "combine_description": combine_desc,
                       "mean_threshold": (mean_threshold
                                          if combine_mode == "mean" else None),
                       "min_blob_px": args.min_blob_px,
                       "watershed": not args.no_watershed,
                       "watershed_min_distance": ws_distance,
                       "min_axis_ratio": args.min_axis_ratio,
                       "patch": patch, "stride": args.stride,
                       "distance_pc": args.distance_pc},
        "git_commit": git_commit,
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__,
                     "tensorflow": tf.__version__},
        "results": {"n_candidates": int(len(candidates)),
                    "mask_px": int(mask.sum()),
                    "prob_max": round(float(prob.max()), 4)},
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\ndone in {time.time() - t0:.0f}s -> {out}/")
    for name in ["probability_map.npy", "probability_map.fits",
                 "binary_mask.npy", "prediction_overlay.png",
                 "candidate_regions.csv", "candidate_regions.reg",
                 "run_metadata.json"]:
        print(f"    {name}")


if __name__ == "__main__":
    main()
