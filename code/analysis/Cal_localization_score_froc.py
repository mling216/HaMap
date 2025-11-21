#!/usr/bin/env python3
"""
CAMELYON-style FROC evaluation from tile-wise probabilities (224x224 tiles).

Inputs per slide (image_name like 'test_001'):
- Slide image: SLIDE_PATH/{image_name}.tif
- Ground truth mask: BASE_TRUTH_DIR/{image_name}_mask.(tif|tiff|png)  (fallback: {image_name}.tif)
- Predictions CSV: PREDICTION_PATH/{image_name}.csv with columns: yhat,x,y
  where x,y are level-0 coordinates divided by 224 (tile indices).

Outputs:
- Prints CAMELYON localization score (avg sensitivity at FP/WSI ∈ {0.25,0.5,1,2,4,8})
- Prints sensitivity at those targets
- Optional FROC plot (set plot=True in run_froc_for_images)

"""
import os
import os.path as osp
import numpy as np
from typing import Dict, List, Tuple
import csv
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
import openslide
from openslide import OpenSlide, PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
import argparse

# -----------------------
# Config (YOUR paths)
# -----------------------
PATCH_SIZE = 224
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing/images/'
BASE_TRUTH_DIR = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing_masks/'
ANNOTATION_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/lesion_annotations_test/'  # not used directly here

# -----------------------
# File helpers
# -----------------------
def prob_filename(image_name: str, prediction_path: str) -> str:
    return osp.join(prediction_path, f'{image_name}.csv')

def gt_mask_filename(image_name: str) -> str:
    for ext in ('.tif', '.tiff', '.png'):
        p = osp.join(BASE_TRUTH_DIR, image_name + '_mask' + ext)
        if osp.exists(p):
            return p
    p = osp.join(BASE_TRUTH_DIR, image_name + '.tif')
    if osp.exists(p):
        return p
    raise FileNotFoundError(f'GT mask for {image_name} not found in {BASE_TRUTH_DIR}')

# -----------------------
# Core utils
# -----------------------
def get_slide_mpp(slide: OpenSlide, fallback_from_objective: bool = True) -> float:
    """Return microns-per-pixel at level 0 for a WSI."""
    mpp_x = slide.properties.get(PROPERTY_NAME_MPP_X, None)
    mpp_y = slide.properties.get(PROPERTY_NAME_MPP_Y, None)
    if mpp_x is not None and mpp_y is not None:
        return (float(mpp_x) + float(mpp_y)) / 2.0
    if fallback_from_objective:
        obj = slide.properties.get('openslide.objective-power', None)
        if obj is not None:
            obj = float(obj)
            approx = {40: 0.25, 20: 0.50, 10: 1.00}
            return approx.get(obj, 10.0 / obj)
    raise ValueError("MPP not found and no reliable fallback from objective power.")

def tile_metrics_for_camelyon(wsi_path: str, tile_size_px: int = PATCH_SIZE, match_radius_um: float = 75.0) -> Dict[str, float]:
    slide = OpenSlide(wsi_path)
    mpp = get_slide_mpp(slide)
    tile_um = tile_size_px * mpp
    r_tiles = max(1, round(match_radius_um / tile_um))
    return {"mpp_um_per_px": mpp, "tile_size_um": tile_um, "match_radius_tiles": r_tiles}

def read_fullres_mask(mask_path: str) -> np.ndarray:
    """Read GT mask as binary numpy array (0/1) at full resolution."""
    ext = osp.splitext(mask_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        try:
            slide = OpenSlide(mask_path)
            w, h = slide.level_dimensions[0]
            rgba = slide.read_region((0, 0), 0, (w, h)).convert('L')
            arr = np.array(rgba)
            return (arr > 0).astype(np.uint8)
        except openslide.OpenSlideUnsupportedFormatError:
            pass
    img = Image.open(mask_path).convert('L')
    arr = np.array(img)
    return (arr > 0).astype(np.uint8)

def downsample_gt_to_tiles(gt_fullres_mask: np.ndarray, tile_size_px: int = PATCH_SIZE, gt_frac: float = 0.5) -> np.ndarray:
    """Aggregate full-res GT mask into a tile grid using fraction threshold."""
    H, W = gt_fullres_mask.shape
    Ht, Wt = H // tile_size_px, W // tile_size_px
    if Ht == 0 or Wt == 0:
        raise ValueError("Mask smaller than one tile; check inputs.")
    gtc = gt_fullres_mask[:Ht*tile_size_px, :Wt*tile_size_px]
    block = gtc.reshape(Ht, tile_size_px, Wt, tile_size_px)
    frac = block.mean(axis=(1, 3))
    gt_tiles = (frac >= gt_frac).astype(np.uint8)
    return gt_tiles

def dilate_gt_for_tolerance(gt_tiles: np.ndarray, r_tiles: int) -> np.ndarray:
    if r_tiles <= 0:
        return gt_tiles.astype(bool)
    se = np.ones((2 * r_tiles + 1, 2 * r_tiles + 1), dtype=bool)
    return ndi.binary_dilation(gt_tiles.astype(bool), se)

def load_prob_tiles_from_csv(csv_path: str) -> np.ndarray:
    """
    Load tile-wise probabilities from a CSV with columns: yhat,x,y
    where (x,y) are tile indices (level 0 coords / 224). Duplicates keep max.
    Returns Ht x Wt float32 in [0,1].
    """
    data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding=None)
    names = [n.lower().strip() for n in data.dtype.names]
    name_to_idx = {n: i for i, n in enumerate(names)}

    def get_col(name: str):
        key = name.lower().strip()
        if key not in name_to_idx:
            raise KeyError(f"Column '{name}' not found in {csv_path}. Found: {names}")
        return np.asarray(data[key])

    yhat = get_col('yhat').astype(float)
    x = np.rint(get_col('x')).astype(int)
    y = np.rint(get_col('y')).astype(int)

    Ht = int(y.max()) + 1 if y.size else 0
    Wt = int(x.max()) + 1 if x.size else 0
    if Ht == 0 or Wt == 0:
        return np.zeros((0, 0), dtype=np.float32)

    grid = np.zeros((Ht, Wt), dtype=np.float32)
    yhat = np.clip(yhat.astype(np.float32), 0.0, 1.0)
    # resolve duplicates using max
    np.maximum.at(grid, (y, x), yhat)
    return grid

def filter_small_pred_regions(det_mask, min_region_tiles):
    """
    det_mask: boolean HxW array of predicted tumor tiles at a given threshold.
    min_region_tiles: components with size <= this are removed as noise,
                      unless all components are <= this, in which case
                      we keep only the largest component.
    """
    if min_region_tiles <= 0:
        return det_mask  # no filtering

    # Label connected components in prediction mask
    labels, num = ndi.label(det_mask.astype(bool),
                            structure=np.ones((3, 3), dtype=int))
    if num == 0:
        return det_mask  # nothing to filter

    # Component sizes (in tiles)
    sizes = ndi.sum(det_mask, labels, index=np.arange(1, num + 1))
    sizes = sizes.astype(int)
    max_size = int(sizes.max())

    # Case 1: largest region is still <= threshold -> keep ONLY the largest region
    if max_size <= min_region_tiles:
        largest_label = int(np.argmax(sizes)) + 1  # labels are 1..num
        return (labels == largest_label)

    # Case 2: keep all regions with size > threshold, drop others
    keep_labels = 1 + np.where(sizes > min_region_tiles)[0]  # convert 0-based idx -> label ids
    return np.isin(labels, keep_labels)

def match_detections_to_lesions(prob_tiles: np.ndarray, gt_tiles: np.ndarray, r_tiles: int,
                                thresholds: np.ndarray = np.linspace(0, 1, 101),
                                min_region_tiles=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    One slide: build detections from prob_tiles >= threshold.
    TP: detections overlapping dilated GT.
    FP: detections outside dilated GT.
    Return thresholds, tps, fps, n_lesions.
    """
    lbl, n_lesions = ndi.label(gt_tiles.astype(bool),
                               structure=np.ones((3, 3), dtype=int))
    dil = dilate_gt_for_tolerance(gt_tiles, r_tiles=r_tiles)
    lbl_dil, _ = ndi.label(dil.astype(bool),
                           structure=np.ones((3, 3), dtype=int))

    tps, fps = [], []
    for th in thresholds:
        det = (prob_tiles >= th)

        # 🔴 NEW: remove tiny predicted regions
        det = filter_small_pred_regions(det, min_region_tiles)

        # Count TP: number of GT lesions hit by at least one predicted region
        hit_labels = set(np.unique(lbl_dil[det])) - {0}
        tp = min(len(hit_labels), n_lesions)

        # Count FP: number of predicted regions (connected components) not overlapping with any GT lesion
        pred_labels, num_pred = ndi.label(det.astype(bool), structure=np.ones((3, 3), dtype=int))
        fp_count = 0
        for region_label in range(1, num_pred + 1):
            region_mask = (pred_labels == region_label)
            # If region does not overlap with any dilated GT, count as FP
            if not np.any(dil[region_mask]):
                fp_count += 1
        tps.append(tp)
        fps.append(fp_count)

    return thresholds, np.asarray(tps, float), np.asarray(fps, float), n_lesions

def aggregate_froc_over_slides(per_slide_results: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = per_slide_results[0]['thresholds']
    total_lesions = sum(d['n_lesions'] for d in per_slide_results)
    tps_sum = np.sum([d['tps'] for d in per_slide_results], axis=0)
    fps_avg = np.mean([d['fps'] for d in per_slide_results], axis=0)
    sensitivity = tps_sum / max(1, total_lesions)
    return thresholds, fps_avg, sensitivity

def camelyon_localization_score(fps_avg: np.ndarray, sensitivity: np.ndarray,
                                fp_targets=(0.25, 0.5, 1, 2, 4, 8)) -> Tuple[float, Dict[float, float]]:
    order = np.argsort(fps_avg)
    x = fps_avg[order]
    y = sensitivity[order]
    uniq_x, idx = np.unique(x, return_index=True)
    uniq_y = y[idx]
    f = interp1d(uniq_x, uniq_y, kind='linear', bounds_error=False,
                 fill_value=(uniq_y[0], uniq_y[-1]))
    sens_at = {fp: float(f(fp)) for fp in fp_targets}
    return float(np.mean(list(sens_at.values()))), sens_at

# -----------------------
# Runner
# -----------------------
def run_froc_for_images(image_names, 
                        prediction_path,
                        outdir,
                        gt_frac=0.5,
                        thresholds=np.linspace(0, 1, 101),
                        plot=False,
                        min_region_tiles=1):

    # ----------------------
    # Prepare output folder + CSV files
    # ----------------------
    os.makedirs(outdir, exist_ok=True)

    per_slide_csv = osp.join(outdir, f"per_slide_results_t{min_region_tiles}.csv")
    per_slide_curves_dir = osp.join(outdir, f"per_slide_curves_t{min_region_tiles}")
    global_csv    = osp.join(outdir, f"global_froc_summary_t{min_region_tiles}.csv")
    
    # Create directory for saving full threshold curves per slide
    os.makedirs(per_slide_curves_dir, exist_ok=True)

    # ----------------------
    # Load already processed slides
    # ----------------------
    processed_slides = set()
    if osp.exists(per_slide_csv):
        try:
            with open(per_slide_csv, "r") as f:
                r = csv.reader(f)
                header = next(r, None)  # skip header
                for row in r:
                    if len(row) > 0:
                        processed_slides.add(row[0])
            print(f"Found {len(processed_slides)} slides already processed.")
        except Exception as e:
            print(f"[WARNING] Could not read existing per-slide CSV: {e}")

    # Initialize per-slide CSV if not already created
    if not osp.exists(per_slide_csv):
        with open(per_slide_csv, "w", newline="") as f:
            w = csv.writer(f)
            # header
            w.writerow([
                "slide",
                "Ht", "Wt",
                "n_lesions",
                "TP_at_th0.5", "FP_at_th0.5",
                "TP_at_th0.9", "FP_at_th0.9",
            ])

    per_slide = []

    # ----------------------
    # Process each slide independently
    # ----------------------
    for image_name in image_names:

        if image_name in processed_slides:
            print(f"Skipping {image_name} — already in CSV.")
            continue

        try:
            print(f"\nProcessing {image_name} ...")
            wsi_path = osp.join(SLIDE_PATH, image_name + '.tif')
            prob_path = prob_filename(image_name, prediction_path)
            gt_mask_path = gt_mask_filename(image_name)

            prob_tiles = load_prob_tiles_from_csv(prob_path)
            gt_full = read_fullres_mask(gt_mask_path)
            gt_tiles = downsample_gt_to_tiles(gt_full, tile_size_px=PATCH_SIZE, gt_frac=gt_frac)

            # Align shapes
            Ht, Wt = prob_tiles.shape
            if gt_tiles.shape != (Ht, Wt):
                Hc, Wc = min(Ht, gt_tiles.shape[0]), min(Wt, gt_tiles.shape[1])
                prob_tiles = prob_tiles[:Hc, :Wc]
                gt_tiles = gt_tiles[:Hc, :Wc]
                Ht, Wt = prob_tiles.shape

            info = tile_metrics_for_camelyon(wsi_path, tile_size_px=PATCH_SIZE)
            r_tiles = int(info['match_radius_tiles'])

            th, tps, fps, n_lesions = match_detections_to_lesions(
                prob_tiles, gt_tiles, r_tiles, thresholds=thresholds,
                min_region_tiles=min_region_tiles
            )

            # Skip slides with zero GT lesions after downsampling
            if n_lesions == 0:
                print(f"Skipping {image_name} — no GT lesions after downsampling.")
                continue

            # ----------------------
            # Append per-slide CSV result
            # ----------------------
            def find_th_idx(v):
                return int(np.argmin(np.abs(th - v)))

            idx05 = find_th_idx(0.5)
            idx09 = find_th_idx(0.9)

            with open(per_slide_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    image_name,
                    Ht, Wt,
                    n_lesions,
                    int(tps[idx05]), int(fps[idx05]),
                    int(tps[idx09]), int(fps[idx09]),
                ])

            # Save full threshold curves for this slide (for later aggregation)
            curve_file = osp.join(per_slide_curves_dir, f"{image_name}_curve.csv")
            curve_df = pd.DataFrame({
                'threshold': th,
                'TP': tps,
                'FP': fps,
                'n_lesions': n_lesions
            })
            curve_df.to_csv(curve_file, index=False)

            # Store full slide results for global FROC (if processing all in one run)
            per_slide.append({
                "thresholds": th,
                "tps": tps,
                "fps": fps,
                "n_lesions": n_lesions,
                "image_name": image_name,
            })

        except Exception as e:
            print(f"[ERROR] Slide {image_name} failed: {e}")
            # continue to next slide safely
            continue

    # ----------------------
    # After all slides → compute global FROC
    # ----------------------
    if not per_slide:
        print("No new slides processed in this run.")
        print("Note: To compute FROC from existing results, you need to re-process all slides")
        print("      because the per-slide CSV doesn't store full threshold curves.")
        print("      Use --batch 0 with an empty output directory, or delete the per_slide CSV first.")
        return None

    thresholds, fps_avg, sensitivity = aggregate_froc_over_slides(per_slide)
    score, sens_table = camelyon_localization_score(fps_avg, sensitivity)

    # ----------------------
    # Append global results CSV
    # ----------------------
    write_header = not osp.exists(global_csv)
    with open(global_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "run_id",
                "num_slides",
                "score",
                "sensitivity_fp_0.25",
                "sensitivity_fp_0.5",
                "sensitivity_fp_1",
                "sensitivity_fp_2",
                "sensitivity_fp_4",
                "sensitivity_fp_8"
            ])
        w.writerow([
            len(per_slide),  # run id = number of slides processed in THIS run
            len(per_slide),  # explicit count
            score,
            sens_table[0.25],
            sens_table[0.5],
            sens_table[1],
            sens_table[2],
            sens_table[4],
            sens_table[8],
        ])

    # ----------------------
    # Plot if requested
    # ----------------------
    if plot:
        plt.figure()
        order = np.argsort(fps_avg)
        plt.plot(fps_avg[order], sensitivity[order], marker='o')
        for x in (0.25, 0.5, 1, 2, 4, 8):
            plt.axvline(x=x, linestyle='--')
        plt.xlabel("Average False Positives per WSI")
        plt.ylabel("Sensitivity")
        plt.title("FROC (CAMELYON-style)")
        plt.grid(True)
        plt.show()

    return {
        "thresholds": thresholds,
        "fps_avg": fps_avg,
        "sensitivity": sensitivity,
        "score": score,
        "sens_at_targets": sens_table,
        "per_slide": per_slide
    }


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAMELYON-style FROC evaluation")
    parser.add_argument('--prediction_path', type=str, required=True,
                        help='Directory containing prediction CSV files (e.g., ./whole_slide_prediction_HaMap/)')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory for FROC results (e.g., ./localization_score_FROC_HaMap)')
    parser.add_argument('--batch', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help='Batch number (1-5) to process slides in batches, or 0 to process all (default: 0)')
    parser.add_argument('--min_region', type=int, default=10,
                        help='Minimum region size (in tiles) to keep as detection (default: 10)')
    parser.add_argument('--plot', action='store_true', help='Plot FROC curve')
    parser.add_argument('--compute_froc', action='store_true',
                        help='Compute global FROC from existing per-slide CSV (use after all batches complete)')
    args = parser.parse_args()

    # get tumor slides only
    df_images = pd.read_csv('../cam16_test_reference.csv')
    df_images = df_images[df_images['type']=='Tumor']
    df_images.sort_values('image_id', inplace=True)

    # Discover slides that have both the WSI and the prediction CSV
    all_image_names = []
    for name in df_images['image_id'].tolist():
        if osp.exists(osp.join(SLIDE_PATH, name + '.tif')) and osp.exists(prob_filename(name, args.prediction_path)):
            all_image_names.append(name)

    print(f"Total slides found with both WSI and prediction CSV: {len(all_image_names)}")
    
    # Split into batches if requested
    if args.batch > 0:
        num_batches = 5
        batch_size = (len(all_image_names) + num_batches - 1) // num_batches
        start_idx = (args.batch - 1) * batch_size
        end_idx = min(args.batch * batch_size, len(all_image_names))
        image_names = all_image_names[start_idx:end_idx]
        print(f"Processing batch {args.batch}/{num_batches}: slides {start_idx+1} to {end_idx} ({len(image_names)} slides)")
    else:
        image_names = all_image_names
        print(f"Processing all {len(image_names)} slides")

    print(f"Prediction path: {args.prediction_path}")
    print(f"Output directory: {args.outdir}")

    # Optional: show tile metrics for the first available slide
    if image_names:
        eg = image_names[0]
        info = tile_metrics_for_camelyon(osp.join(SLIDE_PATH, eg + '.tif'))
        print(f"{eg} tile metrics:", info)

    # Run FROC over the set
    # If in batch mode (batch > 0), skip FROC computation
    # If --compute_froc is set, compute FROC from existing per-slide CSV without processing slides
    if args.compute_froc:
        print("\n" + "="*60)
        print("Computing global FROC from existing per-slide results...")
        print("="*60)
        
        # Load per-slide results
        per_slide_csv = osp.join(args.outdir, f"per_slide_results_t{args.min_region}.csv")
        if not osp.exists(per_slide_csv):
            print(f"ERROR: Per-slide CSV not found at {per_slide_csv}")
            print("Please run slide processing first (without --compute_froc)")
            exit(1)
        
        # Read per-slide data and reconstruct for FROC computation
        # This is a simplified version - we won't have full threshold curves
        # Just report that FROC computation requires re-running with batch=0
        print(f"Found per-slide results at: {per_slide_csv}")
        df_results = pd.read_csv(per_slide_csv)
        print(f"Total slides in CSV: {len(df_results)}")
        print("\nNOTE: Full FROC curve computation requires re-running without batches (--batch 0)")
        print("The per-slide CSV contains TP/FP at specific thresholds (0.5, 0.9) but not the full curve.")
        
    else:
        # Normal processing mode
        skip_global_froc = (args.batch > 0)
        
        results = run_froc_for_images(image_names, 
                                       prediction_path=args.prediction_path,
                                       outdir=args.outdir,
                                       gt_frac=0.5, 
                                       plot=args.plot, 
                                       min_region_tiles=args.min_region)
        
        if skip_global_froc:
            print("\n" + "="*60)
            print("Batch processing complete - per-slide results saved.")
            print("Global FROC metrics NOT computed (requires all batches).")
            print("After all batches finish, run with --batch 0 to compute FROC.")
            print("="*60)

"""
A typical small metastatic focus is hundreds of microns across, so one tile (50 µm) is extremely small relative to a true metastatic lesion.
Thus:
🔴 Any predicted component of size 1 tile
is almost certainly noise
(because the smallest clinical micrometastases are ≈0.2–2 mm = 200–2000 µm).
🟡 2–3 tile components
(100–150 µm) could still be suspicious but often noise in weak models.
🟢 4–10 tile components
start to overlap with tiny micrometastatic clusters.
"""