# HaMap: Habit-Based Fixation Saliency for Whole-Slide Imaging

Pathologists are trained experts of habit, frequently inspecting information-bearing regions during routine whole-slide image (WSI) review. This project investigates whether such human visual habits can be harvested as an *annotation-free* supervisory signal for detecting cancerous tissue.

We introduce **HaMap**, a habit-based fixation saliency map derived from routine WSI readings. Using HaMap to guide an ensemble supervised learning pipeline, we achieve:

- **AUC:** 0.95  
- **Precision:** 0.90 (whole-slide diagnostic classification)  
- **FROC:** 0.08 (tile-level lesion localization)

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

3. GPU Support (optional)
If using PyTorch with CUDA, install the matching version from:
https://pytorch.org/get-started/locally/. Example:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
