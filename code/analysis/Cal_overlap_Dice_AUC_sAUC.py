"""
Cal_overlap_Dice_AUC_sAUC.py

Compute per-slide Dice score, ROC AUC, and shuffled AUC (sAUC) between
CAMELYON16 whole-slide ground-truth masks and tile-level
prediction CSVs.

sAUC (shuffled AUC) uses tumor tiles from other slides as negatives
to test if the model can distinguish this slide's tumor from other slides' tumor.

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
import argparse as ap

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


def calculate_sauc_for_slide(prob_tiles: np.ndarray,
                             gt_tiles: np.ndarray,
                             all_tumor_predictions: dict,
                             current_slide_name: str):
    """
    Compute shuffled AUC (sAUC) for one slide.
    
    Uses tumor tiles from other slides as negative examples to test
    if the model can distinguish this slide's tumor from other slides' tumor.
    
    Parameters
    ----------
    prob_tiles : np.ndarray
        Prediction probabilities for current slide
    gt_tiles : np.ndarray
        Ground truth for current slide
    all_tumor_predictions : dict
        Dictionary mapping slide_name -> (tumor_probs, tumor_gt)
        where tumor_probs are predictions for tumor tiles only
    current_slide_name : str
        Name of current slide (to exclude from negatives)
    
    Returns
    -------
    sauc : float or np.nan
    """
    # Align shapes
    Hc = min(prob_tiles.shape[0], gt_tiles.shape[0])
    Wc = min(prob_tiles.shape[1], gt_tiles.shape[1])
    
    prob = prob_tiles[:Hc, :Wc].astype(float)
    gt = gt_tiles[:Hc, :Wc].astype(np.uint8)
    
    # Get tumor tiles from current slide (positives)
    tumor_mask = (gt == 1)
    if tumor_mask.sum() == 0:
        return np.nan  # No tumor tiles on this slide
    
    current_tumor_probs = prob[tumor_mask]
    
    # Collect tumor predictions from other slides (negatives)
    other_tumor_probs = []
    for slide_name, (other_probs, other_gt) in all_tumor_predictions.items():
        if slide_name != current_slide_name:
            other_tumor_probs.extend(other_probs)
    
    if len(other_tumor_probs) == 0:
        return np.nan  # No other slides with tumor
    
    # Sample negatives to balance with positives (optional, for efficiency)
    # Use same number of negatives as positives, or all if fewer
    n_pos = len(current_tumor_probs)
    other_tumor_probs = np.array(other_tumor_probs)
    
    if len(other_tumor_probs) > n_pos:
        # Randomly sample to match positive count
        np.random.seed(42)  # For reproducibility
        neg_indices = np.random.choice(len(other_tumor_probs), size=n_pos, replace=False)
        other_tumor_probs = other_tumor_probs[neg_indices]
    
    # Construct y_true and y_score for sAUC
    # Positives: current slide tumor (label=1)
    # Negatives: other slides tumor (label=0)
    y_true = np.concatenate([
        np.ones(len(current_tumor_probs)),
        np.zeros(len(other_tumor_probs))
    ])
    
    y_score = np.concatenate([
        current_tumor_probs,
        other_tumor_probs
    ])
    
    # Calculate sAUC
    try:
        sauc = roc_auc_score(y_true, y_score)
    except:
        sauc = np.nan
    
    return sauc


# --------------------
# Main loop
# --------------------

def run_dice_auc_for_images(image_names, prediction_path, out_csv):
    """
    Compute per-slide Dice, AUC, and sAUC and write to out_csv.
    
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

    # ===== PASS 1: Load all predictions and collect tumor tiles =====
    print("\nPass 1: Loading all predictions and collecting tumor tiles...")
    all_tumor_predictions = {}  # slide_name -> (tumor_probs, tumor_gt)
    all_slide_data = {}  # slide_name -> (prob_tiles, gt_tiles, mask_path, csv_path)
    
    for slide_name in image_names:
        try:
            mask_path = gt_mask_filename(slide_name)
            csv_path = osp.join(prediction_path, f"{slide_name}.csv")
            
            # Load data
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
            
            # Store all data
            all_slide_data[slide_name] = (prob_tiles, gt_tiles, mask_path, csv_path)
            
            # Extract tumor tiles for sAUC
            tumor_mask = (gt_tiles == 1)
            if tumor_mask.sum() > 0:
                tumor_probs = prob_tiles[tumor_mask].ravel()
                tumor_gt = gt_tiles[tumor_mask].ravel()
                all_tumor_predictions[slide_name] = (tumor_probs, tumor_gt)
                
        except Exception as e:
            print(f"[ERROR] Failed to load {slide_name}: {e}")
            continue
    
    print(f"Loaded {len(all_slide_data)} slides, {len(all_tumor_predictions)} with tumor tiles")

    # ===== PASS 2: Calculate metrics and write results =====
    print("\nPass 2: Calculating metrics...")
    write_header = not osp.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "slide",
                "Ht", "Wt",
                "n_tumor_tiles",
                "dice_at_th0.5",
                "auc_slide",
                "sauc_slide"
            ])

        for slide_name in image_names:
            if slide_name not in all_slide_data:
                continue  # Skip slides that failed to load
                
            try:
                print(f"\nProcessing {slide_name} ...")
                
                prob_tiles, gt_tiles, mask_path, csv_path = all_slide_data[slide_name]

                # 3) Metrics
                dice, auc = dice_and_auc_for_slide(prob_tiles, gt_tiles,
                                                   dice_threshold=DICE_THRESH)
                
                # Calculate sAUC
                sauc = calculate_sauc_for_slide(prob_tiles, gt_tiles,
                                               all_tumor_predictions,
                                               slide_name)

                Ht = prob_tiles.shape[0]
                Wt = prob_tiles.shape[1]
                n_tumor_tiles = int(gt_tiles.sum())

                # Dice: only meaningful on tumor slides
                if n_tumor_tiles == 0:
                    dice_to_write = np.nan   # normal slide → no GT tumor
                else:
                    dice_to_write = dice

                print(f"  slide={slide_name}  tumor_tiles={n_tumor_tiles}  "
                    f"Dice={dice_to_write}  AUC={auc}  sAUC={sauc}")

                # 4) Write row
                w.writerow([
                    slide_name,
                    Ht, Wt,
                    n_tumor_tiles,
                    dice,
                    auc,
                    sauc
                ])

            except Exception as e:
                print(f"[ERROR] Slide {slide_name} failed: {e}")
                continue


if __name__ == "__main__":
    parser = ap.ArgumentParser(description='Calculate Dice, AUC, and sAUC metrics for whole slide predictions')
    parser.add_argument('--prediction_path', type=str, required=True,
                        help='Directory containing prediction CSV files (e.g., ./whole_slide_prediction_HaMap)')
    parser.add_argument('--out_csv', type=str, required=True,
                        help='Output CSV file path (e.g., overlap_metrics_dice_auc_sauc_HaMap.csv)')
    
    args = parser.parse_args()
    
    # Use cam16_test_reference.csv to get tumor vs normal
    ref_df = pd.read_csv('../cam16_test_reference.csv')

    # All slides (for AUC)
    all_slides = ref_df['image_id'].tolist()

    # Tumor-only slides (for Dice, if you want a tumor-only summary)
    tumor_slides = ref_df[ref_df['type'] == 'Tumor']['image_id'].tolist()

    print(f"Total slides in reference: {len(all_slides)}")
    print(f"Tumor slides: {len(tumor_slides)}")
    print(f"Prediction path: {args.prediction_path}")
    print(f"Output CSV: {args.out_csv}")

    # Option 1: run on all slides; Dice will be NaN for normals, AUC valid for all
    run_dice_auc_for_images(all_slides, args.prediction_path, args.out_csv)
