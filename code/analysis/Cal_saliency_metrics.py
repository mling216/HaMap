"""
Cal_saliency_metrics.py

Compute per-slide saliency metrics between CAMELYON16 whole-slide 
ground-truth tumor masks and tile-level prediction CSVs.

Metrics computed:
- NSS (Normalized Scanpath Saliency): measures how well predictions align with GT tumor locations
- KL (Kullback-Leibler Divergence): distributional distance (lower is better)
- SIM (Similarity): overlap between normalized distributions (higher is better)
- IG (Information Gain): improvement over center-bias baseline (higher is better)

These are information-theoretic metrics from visual saliency literature,
adapted for pathology tumor detection.

Assumptions
----------
- Each slide has a prediction CSV:
    PREDICTION_PATH / {slide_name}.csv
  with columns: yhat,x,y
  where x,y are tile indices at level 0 / PATCH_SIZE.

- Each slide has a GT mask image:
    BASE_TRUTH_DIR / {slide_name}_mask.tif

- PATCH_SIZE is the tile size used when generating predictions.
"""

import os
import os.path as osp
import csv
import argparse as ap

import numpy as np
import pandas as pd
from imageio.v2 import imread


# --------------------
# Config
# --------------------

PATCH_SIZE = 224

SLIDE_PATH      = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing/images/'
BASE_TRUTH_DIR  = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing_masks/'

# Minimum fraction of tumor pixels in a tile to call it positive
GT_FRAC = 0.5

# Small epsilon to avoid log(0) and division by zero
EPSILON = 1e-10


# --------------------
# Helpers
# --------------------

def gt_mask_filename(slide_name: str) -> str:
    """
    Return full path to GT mask image for a given slide.
    """
    fname = f"{slide_name}_mask.tif"
    path = osp.join(BASE_TRUTH_DIR, fname)
    if not osp.exists(path):
        alt = osp.join(BASE_TRUTH_DIR, f"{slide_name}.tif")
        if osp.exists(alt):
            return alt
    return path


def read_fullres_mask(mask_path: str) -> np.ndarray:
    """
    Read the full-resolution tumor mask as a binary numpy array.
    """
    if not osp.exists(mask_path):
        raise FileNotFoundError(mask_path)
    arr = imread(mask_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def downsample_gt_to_tiles(gt_full: np.ndarray,
                           tile_size_px: int = PATCH_SIZE,
                           gt_frac: float = GT_FRAC) -> np.ndarray:
    """
    Downsample full-res GT mask to tile grid.
    Returns fraction of tumor pixels per tile (continuous values [0,1]).
    """
    H, W = gt_full.shape
    Ht = H // tile_size_px
    Wt = W // tile_size_px

    gt_cropped = gt_full[:Ht * tile_size_px, :Wt * tile_size_px]

    gt_tiles = gt_cropped.reshape(
        Ht, tile_size_px,
        Wt, tile_size_px
    )

    # Return continuous tumor fraction (not binary)
    tumor_frac = gt_tiles.mean(axis=(1, 3))
    return tumor_frac.astype(np.float32)


def load_prob_tiles_from_csv(csv_path: str) -> np.ndarray:
    """
    Load tile probabilities from CSV with columns yhat,x,y
    into a 2D array prob_tiles[y, x].
    """
    if not osp.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if not {'yhat', 'x', 'y'}.issubset(df.columns):
        raise ValueError(f"{csv_path} must have columns yhat,x,y")

    max_x = int(df['x'].max())
    max_y = int(df['y'].max())

    Ht = max_y + 1
    Wt = max_x + 1

    prob_tiles = np.zeros((Ht, Wt), dtype=np.float32)

    for _, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        prob_tiles[y, x] = float(row['yhat'])

    return prob_tiles


# --------------------
# Saliency Metrics
# --------------------

def calculate_nss(pred_map: np.ndarray, gt_map: np.ndarray) -> float:
    """
    NSS (Normalized Scanpath Saliency)
    
    Normalizes prediction map to zero mean and unit variance,
    then averages values at ground truth tumor locations.
    
    Parameters
    ----------
    pred_map : 2D array of prediction probabilities
    gt_map : 2D array of ground truth tumor fractions [0,1]
    
    Returns
    -------
    nss : float, higher is better (>1 is above chance)
    """
    # Align shapes
    Hc = min(pred_map.shape[0], gt_map.shape[0])
    Wc = min(pred_map.shape[1], gt_map.shape[1])
    pred = pred_map[:Hc, :Wc].astype(float)
    gt = gt_map[:Hc, :Wc].astype(float)
    
    # Normalize prediction to zero mean, unit variance
    pred_mean = pred.mean()
    pred_std = pred.std()
    
    if pred_std < EPSILON:
        return np.nan  # No variation in predictions
    
    pred_norm = (pred - pred_mean) / pred_std
    
    # Get fixation locations (where GT has tumor)
    fixation_mask = gt > 0.5  # Threshold for tumor presence
    
    if fixation_mask.sum() == 0:
        return np.nan  # No tumor in GT
    
    # Average normalized saliency at fixation points
    nss = pred_norm[fixation_mask].mean()
    
    return float(nss)


def calculate_kl(pred_map: np.ndarray, gt_map: np.ndarray) -> float:
    """
    KL (Kullback-Leibler Divergence)
    
    Measures distributional distance between prediction and GT.
    Both maps are normalized to sum to 1 (probability distributions).
    
    Parameters
    ----------
    pred_map : 2D array of prediction probabilities
    gt_map : 2D array of ground truth tumor fractions
    
    Returns
    -------
    kl : float, LOWER is better (0 = identical distributions)
    """
    # Align shapes
    Hc = min(pred_map.shape[0], gt_map.shape[0])
    Wc = min(pred_map.shape[1], gt_map.shape[1])
    pred = pred_map[:Hc, :Wc].astype(float).ravel()
    gt = gt_map[:Hc, :Wc].astype(float).ravel()
    
    # Normalize to probability distributions
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    if pred_sum < EPSILON or gt_sum < EPSILON:
        return np.nan
    
    P = pred / pred_sum  # Predicted distribution
    Q = gt / gt_sum      # Ground truth distribution
    
    # Add epsilon to avoid log(0)
    P = P + EPSILON
    Q = Q + EPSILON
    
    # Re-normalize after adding epsilon
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # KL(Q || P) = sum(Q * log(Q / P))
    kl = np.sum(Q * np.log(Q / P))
    
    return float(kl)


def calculate_sim(pred_map: np.ndarray, gt_map: np.ndarray) -> float:
    """
    SIM (Similarity)
    
    Sums the minimum values between normalized prediction and GT distributions.
    
    Parameters
    ----------
    pred_map : 2D array of prediction probabilities
    gt_map : 2D array of ground truth tumor fractions
    
    Returns
    -------
    sim : float in [0, 1], HIGHER is better (1 = identical)
    """
    # Align shapes
    Hc = min(pred_map.shape[0], gt_map.shape[0])
    Wc = min(pred_map.shape[1], gt_map.shape[1])
    pred = pred_map[:Hc, :Wc].astype(float).ravel()
    gt = gt_map[:Hc, :Wc].astype(float).ravel()
    
    # Normalize to probability distributions
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    if pred_sum < EPSILON or gt_sum < EPSILON:
        return np.nan
    
    P = pred / pred_sum
    Q = gt / gt_sum
    
    # SIM = sum(min(P, Q))
    sim = np.minimum(P, Q).sum()
    
    return float(sim)


def calculate_center_bias_baseline(shape: tuple) -> np.ndarray:
    """
    Create a center-bias baseline saliency map (2D Gaussian centered on image).
    
    Parameters
    ----------
    shape : tuple (H, W)
        Shape of the map
    
    Returns
    -------
    baseline : 2D array of same shape, normalized to sum to 1
    """
    H, W = shape
    
    # Create coordinate grids
    y = np.arange(H)
    x = np.arange(W)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    
    # Center coordinates
    cy = H / 2.0
    cx = W / 2.0
    
    # Standard deviation = 1/4 of image dimension (covers most of image)
    sigma_y = H / 4.0
    sigma_x = W / 4.0
    
    # 2D Gaussian
    gaussian = np.exp(-((yy - cy)**2 / (2 * sigma_y**2) + 
                        (xx - cx)**2 / (2 * sigma_x**2)))
    
    # Normalize to sum to 1
    baseline = gaussian / (gaussian.sum() + EPSILON)
    
    return baseline.astype(np.float32)


def calculate_ig(pred_map: np.ndarray, gt_map: np.ndarray) -> float:
    """
    IG (Information Gain)
    
    Measures how much better the prediction is than a center-bias baseline.
    IG = KL(GT || baseline) - KL(GT || prediction)
    
    Parameters
    ----------
    pred_map : 2D array of prediction probabilities
    gt_map : 2D array of ground truth tumor fractions
    
    Returns
    -------
    ig : float, HIGHER is better (positive means better than baseline)
    """
    # Align shapes
    Hc = min(pred_map.shape[0], gt_map.shape[0])
    Wc = min(pred_map.shape[1], gt_map.shape[1])
    pred = pred_map[:Hc, :Wc].astype(float).ravel()
    gt = gt_map[:Hc, :Wc].astype(float).ravel()
    
    # Normalize GT to distribution
    gt_sum = gt.sum()
    if gt_sum < EPSILON:
        return np.nan
    
    Q = gt / gt_sum
    
    # Normalize prediction to distribution
    pred_sum = pred.sum()
    if pred_sum < EPSILON:
        return np.nan
    
    P_pred = pred / pred_sum
    
    # Create center-bias baseline
    baseline = calculate_center_bias_baseline((Hc, Wc)).ravel()
    baseline_sum = baseline.sum()
    P_base = baseline / baseline_sum
    
    # Add epsilon to all distributions
    Q = Q + EPSILON
    P_pred = P_pred + EPSILON
    P_base = P_base + EPSILON
    
    # Re-normalize
    Q = Q / Q.sum()
    P_pred = P_pred / P_pred.sum()
    P_base = P_base / P_base.sum()
    
    # KL divergences
    kl_base = np.sum(Q * np.log(Q / P_base))
    kl_pred = np.sum(Q * np.log(Q / P_pred))
    
    # Information Gain
    ig = kl_base - kl_pred
    
    return float(ig)


# --------------------
# Main loop
# --------------------

def run_saliency_metrics_for_images(image_names, prediction_path, out_csv):
    """
    Compute per-slide saliency metrics and write to out_csv.
    
    Parameters
    ----------
    image_names : list of str
        Slide names to process
    prediction_path : str
        Directory containing prediction CSV files
    out_csv : str
        Output CSV file path
    """
    # Create output directory if needed
    out_dir = osp.dirname(out_csv)
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    write_header = not osp.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "slide",
                "Ht", "Wt",
                "n_tumor_tiles",
                "NSS",
                "KL",
                "SIM",
                "IG"
            ])

        for slide_name in image_names:
            try:
                print(f"\nProcessing {slide_name} ...")

                # Load data
                mask_path = gt_mask_filename(slide_name)
                csv_path = osp.join(prediction_path, f"{slide_name}.csv")
                
                gt_full = read_fullres_mask(mask_path)
                gt_tiles = downsample_gt_to_tiles(gt_full,
                                                  tile_size_px=PATCH_SIZE,
                                                  gt_frac=GT_FRAC)
                prob_tiles = load_prob_tiles_from_csv(csv_path)

                # Align shapes
                Hc = min(prob_tiles.shape[0], gt_tiles.shape[0])
                Wc = min(prob_tiles.shape[1], gt_tiles.shape[1])
                prob_tiles = prob_tiles[:Hc, :Wc]
                gt_tiles = gt_tiles[:Hc, :Wc]

                # Count tumor tiles (using binary threshold for reporting)
                n_tumor_tiles = int((gt_tiles > 0.5).sum())

                # Calculate metrics
                nss = calculate_nss(prob_tiles, gt_tiles)
                kl = calculate_kl(prob_tiles, gt_tiles)
                sim = calculate_sim(prob_tiles, gt_tiles)
                ig = calculate_ig(prob_tiles, gt_tiles)

                print(f"  slide={slide_name}  tumor_tiles={n_tumor_tiles}")
                print(f"  NSS={nss:.4f}  KL={kl:.4f}  SIM={sim:.4f}  IG={ig:.4f}")

                # Write row
                w.writerow([
                    slide_name,
                    Hc, Wc,
                    n_tumor_tiles,
                    nss,
                    kl,
                    sim,
                    ig
                ])

            except Exception as e:
                print(f"[ERROR] Slide {slide_name} failed: {e}")
                continue

    print(f"\n✓ Results written to {out_csv}")


if __name__ == "__main__":
    parser = ap.ArgumentParser(
        description='Calculate saliency metrics (NSS, KL, SIM, IG) for whole slide predictions'
    )
    parser.add_argument('--prediction_path', type=str, required=True,
                        help='Directory containing prediction CSV files')
    parser.add_argument('--out_csv', type=str, required=True,
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Use cam16_test_reference.csv to get tumor slides only
    # (Saliency metrics are undefined for normal slides with no tumor)
    ref_df = pd.read_csv('../cam16_test_reference.csv')
    tumor_slides = ref_df[ref_df['type'] == 'Tumor']['image_id'].tolist()

    print(f"Total slides in reference: {len(ref_df)}")
    print(f"Tumor slides (processing): {len(tumor_slides)}")
    print(f"Prediction path: {args.prediction_path}")
    print(f"Output CSV: {args.out_csv}")
    print(f"\nNote: Only processing tumor slides (normal slides have NaN metrics)")

    # Run on tumor slides only
    run_saliency_metrics_for_images(tumor_slides, args.prediction_path, args.out_csv)
