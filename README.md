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

