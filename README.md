# HaMap: Habit-Based Fixation Saliency for Whole-Slide Imaging

Pathologists are trained experts of habit, frequently inspecting information-bearing regions during routine whole-slide image (WSI) review. This project investigates whether such human visual habits can be harvested as an *annotation-free* supervisory signal for detecting cancerous tissue.

We introduce **HaMap**, a habit-based fixation saliency map derived from routine WSI readings. Using HaMap to guide an ensemble supervised learning pipeline, we achieve:

- **AUC:** 0.95  
- **Precision:** 0.90 (whole-slide diagnostic classification)  
- **FROC:** 0.16 (tile-level lesion localization)

We validate HaMap’s reliability using four independent experiments. The results show:

- Fixation-derived saliency contains informative signals capable of achieving **higher diagnostic precision (0.90)** than pathologists alone (~0.75).
- HaMap outperforms viewport-based and weakly supervised learning baselines.
- **HaMap++** provides additional gains by fine-tuning weakly supervised models for slide-level classification.
- Our annotation-free data collection method integrates seamlessly with daily clinical workflows and can scale across institutions.

While whole-slide classification performance is strong, tile-level localization remains an area for further improvement.

This repository will contain code, datasets, and pipelines for generating HaMap and HaMap++, as well as tools for evaluating saliency-guided diagnostic algorithms.

## Environment Setup

This project was developed and tested using Python X.Y and Conda.
To reproduce the environment:
1. Create the environment from the YAML file
```
conda env create -f environment.yml
conda activate hamap
```

2. (Optional) Install packages via pip
```
pip install -r requirements.txt
```

---

## Usage Pipeline

After setting up the environment, follow these steps to reproduce the HaMap pipeline:

### 1. Extract Tiles
Extract tissue tiles from whole-slide images using scripts in `code/preprocessing/tilling/`.

### 2. Stain Normalize Tiles
Normalize the extracted tiles using `code/preprocessing/stain_norm/` (see Stain Normalization section below).

### 3. Train and Test Models
Train and test models using scripts in `code/models/training/` (see Train and Test a Model section below).

### 4. Run Analyses
Perform analyses using scripts in `code/analysis/`. Examples:
- `Cal_overlap_Dice_AUC_sAUC.py` - Calculate overlap, Dice, AUC, and sAUC metrics
- `Analyze_FROC_score.ipynb` - Analyze FROC scores

---

## Stain Normalization Environment (stainenv)

Stain normalization of tissue tiles requires a separate Conda environment due to specific dependencies (e.g., staintools, spams).

To set up the `stainenv` environment:

1. Create the environment from the provided YAML file:
	```
	conda env create -f stainenv.yml
	conda activate stainenv
	```

2. Run stain normalization scripts (batch split recommended):
	```
	python stain_normalize_tiles_split.py -d <tile_dir> -n <num_batches> -b <batch_num>
	```
	Example:
	```
	python stain_normalize_tiles_split.py -d train/tumor/fixation_reduction_0.5_s6 -n 10 -b 1
	```
	This splits the tiles into 10 batches and processes batch 1. Adjust `-b` for other batches.

**Note:**
- The `stainenv.yml` file includes all required dependencies for stain normalization.
- If you encounter issues with `spams`, the scripts will fall back to Macenko normalization (see code comments).

---

## Train and Test a Model

To train and test a model:

```bash
cd code/models/train-test/
python modelDense256_vgg19_train_test.py -d tumor_0.3_s16
```

Where:
- `-d` specifies the directory name for the training data
- `0.3` represents the HaMap threshold level
- `s16` indicates random seed 16

---

## Results

### Analysis Notebooks
Key analysis notebooks are available in `code/analysis/`:
- `Analyze_FROC_score.ipynb` - FROC score analysis
- `Analyze_ablation_results.ipynb` - Ablation study results
- `slide_classification.ipynb` - Slide-level classification analysis
- `eyetracking/` - Eye-tracking visualization notebooks

### Result Figures
Result plots are organized in `figures/`:

**HaMap Results** (`figures/hamap_results_plots/`):
- `AUC_compare_voting*.png` - AUC comparisons across voting strategies
- `exp1_cohen_kappa.png` - Inter-rater agreement analysis
- `exp1_result_voting_f1.png` - Voting results with F1 scores
- `exp1_compare_wt_baseline_*.png` - Comparison with baseline methods
- `exp2_compare_gaze_maps_*.png` - Gaze map comparison metrics
- `fixation_iou_groundtruth.png` - Fixation overlap with ground truth
- `num_fixation_tiles.png` - Fixation tile distribution

**HaMap++ Results** (`figures/hamap_pp_results_plots/`):
- `expX_model_comparison*.png` - Model performance comparisons
- `mask_comparison_*.png` - Mask precision/recall analysis
- `CLAM_vs_supervised_*.png` - CLAM vs supervised learning comparison
- `expX_normal_gaze_proportion_comparison.png` - Normal gaze proportion analysis

**Eye-tracking Visualizations** (`figures/eyetracking/`):
- Thumbnail and PFMap visualizations with ground truth overlays

---

