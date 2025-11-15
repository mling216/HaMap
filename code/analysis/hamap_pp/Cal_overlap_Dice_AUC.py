"""
Cal_overlap_Dice_AUC.py

Compute per-slide Dice score and ROC AUC between
CAMELYON16 whole-slide ground-truth masks and tile-level
prediction CSVs.

Assumptions
----------
- Each slide has a prediction CSV:
    PREDICTION_PATH / {slide_name}.csv
  with columns: yhat,x,y
  where x,y are tile indices at level 0 / PATCH_SIZE.

- Each slide has a GT mask image:
    BASE_TRUTH_DIR / {slide_name}_mask.tif
  (Adjust gt_mask_filename() if your pattern is different.)

- PATCH_SIZE is the tile size used when generating predictions.
"""

import os
import os.path as osp
import csv

import numpy as np
import pandas as pd
from imageio.v2 import imread
from sklearn.metrics import roc_auc_score


# --------------------
# Config
# --------------------

PATCH_SIZE = 224

SLIDE_PATH      = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing/images/'
BASE_TRUTH_DIR  = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing_masks/'
PREDICTION_PATH = './whole_slide_prediction/'

OUT_CSV = './results_Dice_AUC/overlap_metrics_dice_auc.csv'

# Minimum fraction of tumor pixels in a tile to call it positive
GT_FRAC = 0.5

# Probability threshold for Dice
DICE_THRESH = 0.5


# --------------------
# Helpers
# --------------------

def gt_mask_filename(slide_name: str) -> str:
    """
    Return full path to GT mask image for a given slide.

    Adjust this if your mask naming is different.
    Common patterns:
        'test_001_mask.tif'
        'test_001.tif'
    """
    # Example: test_001_mask.tif
    fname = f"{slide_name}_mask.tif"
    path = osp.join(BASE_TRUTH_DIR, fname)
    if not osp.exists(path):
        # Try alternative: same name as slide
        alt = osp.join(BASE_TRUTH_DIR, f"{slide_name}.tif")
        if osp.exists(alt):
            return alt
    return path


def read_fullres_mask(mask_path: str) -> np.ndarray:
    """
    Read the full-resolution tumor mask as a binary numpy array.
    Assumes tumor pixels > 0.
    """
    if not osp.exists(mask_path):
        raise FileNotFoundError(mask_path)
    arr = imread(mask_path)
    # If multi-channel, convert to single channel
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def downsample_gt_to_tiles(gt_full: np.ndarray,
                           tile_size_px: int = PATCH_SIZE,
                           gt_frac: float = GT_FRAC) -> np.ndarray:
    """
    Downsample full-res GT mask to tile grid.

    Parameters
    ----------
    gt_full : 2D np.ndarray, binary
    tile_size_px : int
        Tile size in pixels at level 0.
    gt_frac : float
        Minimum fraction of tumor pixels in a tile to mark tile as positive.

    Returns
    -------
    gt_tiles : 2D np.ndarray of shape (Ht, Wt), dtype uint8, values {0,1}
    """
    H, W = gt_full.shape
    Ht = H // tile_size_px
    Wt = W // tile_size_px

    # crop to multiple of tile_size_px
    gt_cropped = gt_full[:Ht * tile_size_px, :Wt * tile_size_px]

    # reshape into (Ht, tile_size, Wt, tile_size)
    gt_tiles = gt_cropped.reshape(
        Ht, tile_size_px,
        Wt, tile_size_px
    )

    # fraction of tumor pixels in each tile
    tumor_frac = gt_tiles.mean(axis=(1, 3))
    return (tumor_frac >= gt_frac).astype(np.uint8)


def load_prob_tiles_from_csv(csv_path: str) -> np.ndarray:
    """
    Load tile probabilities from CSV with columns yhat,x,y
    into a 2D array prob_tiles[y, x].

    Missing tiles are filled with 0.
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


def dice_and_auc_for_slide(prob_tiles: np.ndarray,
                           gt_tiles: np.ndarray,
                           dice_threshold: float = DICE_THRESH):
    """
    Compute Dice (at a fixed threshold) and ROC AUC for one slide.

    Returns
    -------
    dice : float or np.nan
    auc  : float or np.nan
    """
    # Align shapes by cropping to common area
    Hc = min(prob_tiles.shape[0], gt_tiles.shape[0])
    Wc = min(prob_tiles.shape[1], gt_tiles.shape[1])

    prob = prob_tiles[:Hc, :Wc].astype(float)
    gt = gt_tiles[:Hc, :Wc].astype(np.uint8)

    y_true = gt.ravel()
    y_score = prob.ravel()

    # ----- Dice -----
    y_pred = (y_score >= dice_threshold).astype(np.uint8)
    intersection = np.logical_and(y_pred == 1, y_true == 1).sum()
    pred_pos = (y_pred == 1).sum()
    gt_pos = (y_true == 1).sum()

    if pred_pos + gt_pos == 0:
        dice = np.nan  # no positives anywhere
    else:
        dice = 2.0 * intersection / float(pred_pos + gt_pos)

    # ----- AUC -----
    if y_true.max() == y_true.min():
        auc = np.nan  # all same class, AUC undefined
    else:
        auc = roc_auc_score(y_true, y_score)

    return dice, auc


# --------------------
# Main loop
# --------------------

def run_dice_auc_for_images(image_names):
    """
    Compute per-slide Dice and AUC and write to OUT_CSV.
    """

    write_header = not osp.exists(OUT_CSV)

    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "slide",
                "Ht", "Wt",
                "n_tumor_tiles",
                "dice_at_th0.5",
                "auc_slide"
            ])

        for slide_name in image_names:
            try:
                print(f"\nProcessing {slide_name} ...")

                # 1) Paths
                mask_path = gt_mask_filename(slide_name)
                csv_path = osp.join(PREDICTION_PATH, f"{slide_name}.csv")

                # 2) Load data
                gt_full = read_fullres_mask(mask_path)
                gt_tiles = downsample_gt_to_tiles(gt_full,
                                                  tile_size_px=PATCH_SIZE,
                                                  gt_frac=GT_FRAC)
                prob_tiles = load_prob_tiles_from_csv(csv_path)

                # 3) Metrics
                dice, auc = dice_and_auc_for_slide(prob_tiles, gt_tiles,
                                                   dice_threshold=DICE_THRESH)

                Ht = prob_tiles.shape[0]
                Wt = prob_tiles.shape[1]
                n_tumor_tiles = int(gt_tiles.sum())

                # Dice: only meaningful on tumor slides
                if n_tumor_tiles == 0:
                    dice_to_write = np.nan   # normal slide → no GT tumor
                else:
                    dice_to_write = dice

                print(f"  slide={slide_name}  tumor_tiles={n_tumor_tiles}  "
                    f"Dice={dice_to_write}  AUC={auc}")

                # 4) Write row
                w.writerow([
                    slide_name,
                    Ht, Wt,
                    n_tumor_tiles,
                    dice,
                    auc
                ])

            except Exception as e:
                print(f"[ERROR] Slide {slide_name} failed: {e}")
                continue


if __name__ == "__main__":
    # Use cam16_test_reference.csv to get tumor vs normal
    ref_df = pd.read_csv('../cam16_test_reference.csv')

    # All slides (for AUC)
    all_slides = ref_df['image_id'].tolist()

    # Tumor-only slides (for Dice, if you want a tumor-only summary)
    tumor_slides = ref_df[ref_df['type'] == 'Tumor']['image_id'].tolist()

    print(f"Total slides in reference: {len(all_slides)}")
    print(f"Tumor slides: {len(tumor_slides)}")

    # Option 1: run on all slides; Dice will be NaN for normals, AUC valid for all
    run_dice_auc_for_images(all_slides)
