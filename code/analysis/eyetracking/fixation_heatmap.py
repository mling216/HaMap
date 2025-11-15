import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageDraw
from ast import literal_eval
import openslide

import cv2
from skimage import color                       # rgb2lab, lab2rgb
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes

slide_path = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing/images/'
slide_mask_path = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing_masks/'
fixation_heatmap_path = '/fs/ess/PAS1575/Dataset/new_data/fixation_heatmaps/fixation_reduction'

down_scale = 32 # default value that needs to be adjusted

# # get slide tissue threshold from a set of files
# df_thresholds = pd.DataFrame()
# for i in range(1,11):
#     df_thresholds = pd.concat([pd.read_csv(f'../non_overlap_tiles/test_slides_tissue_thresholds_{i}_of_10.csv'), df_thresholds], ignore_index=True)
# df_thresholds.head()

# get list of fixation reduction heatmap
list_fixation_heatmap = os.listdir(fixation_heatmap_path)
list_fixation_heatmap = [f for f in list_fixation_heatmap if f.endswith('.npy')]
print(f'Number of fixation heatmaps: {len(list_fixation_heatmap)}')

# build an image_id to slide_name dictionary
df_image = pd.read_csv('../experimentData/imageNameMappingWithTumorRatio.csv')

heatmap_dict = {}
for f in list_fixation_heatmap:
    C_name = f.split('_')[0]
    heatmap_dict[C_name] = df_image[df_image['imageID'] == C_name]['imageName'].values[0]

print(list(heatmap_dict.items())[:5])

# list_fixation_heatmap = ['C9_P9.npy']

# Tissue visible; glass white; heatmap only on tissue
# Tumor mask contours overlaid in BLUE

gt_color = (255, 0, 0)   # this is RGB

for example_heatmap in list_fixation_heatmap: # 'C9_P9.npy'

    # Tissue/background thresholds
    L_thresh, C_thresh = 92.0, 8.0
    se = np.ones((3,3), dtype=bool)
    open_iters, close_iters = 2, 2

    # Value→hue sweep params
    hue_start, hue_end = 220.0, 60.0
    kC_lo, kC_hi = 10.0, 80.0
    dL_hot_hi = +12.0
    nonlin_pow = 0.5
    tissue_lighten = 1.10

    # ====== LOAD SLIDE + HEATMAP ======
    imageID = example_heatmap.split('_')[0]
    slide_name = heatmap_dict[imageID]
    print(f'Example heatmap: {example_heatmap}, slide name: {slide_name}')

    slide_fp = os.path.join(slide_path, slide_name + '.tif')
    if not os.path.exists(slide_fp):
        print(f'Slide file not found: {slide_fp}')
        continue

    with openslide.OpenSlide(slide_fp) as slide:
        nx, ny = slide.dimensions
        thumb = slide.get_thumbnail((nx // down_scale, ny // down_scale)).convert("RGB")
    thumbnail = np.asarray(thumb).astype(np.float32) / 255.0
    H = np.load(os.path.join(fixation_heatmap_path, example_heatmap)).astype(np.float32)

    # ====== TISSUE MASK ======
    lab_thumb = color.rgb2lab(thumbnail)
    L_t, A_t, B_t = lab_thumb[...,0], lab_thumb[...,1], lab_thumb[...,2]
    C_t = np.sqrt(A_t**2 + B_t**2)
    is_background = (L_t > L_thresh) & (C_t < C_thresh)
    tissue_mask = ~is_background
    tissue_mask = binary_opening(tissue_mask, structure=se, iterations=open_iters)
    tissue_mask = binary_closing(tissue_mask, structure=se, iterations=close_iters)
    tissue_mask = binary_fill_holes(tissue_mask).astype(bool)

    H *= tissue_mask.astype(np.float32)

    # ====== VALUE SCALING ======
    pos = H[H > 0]
    if pos.size == 0:
        raise ValueError("Heatmap has no positive values after masking.")
    low, high = np.percentile(pos, 5), np.percentile(pos, 99.5)
    norm = mpl.colors.PowerNorm(gamma=0.4, vmin=low, vmax=high)
    Hnorm = np.clip((H - low) / max(1e-6, (high - low)), 0, 1)
    fix_ms = (H > 0).astype(np.float32)
    Hboost = Hnorm**nonlin_pow

    # ====== LAB BLEND ======
    L, A, B = L_t.copy(), A_t.copy(), B_t.copy()
    L[tissue_mask] = np.clip(L[tissue_mask] * tissue_lighten, 0, 100)

    hue_deg_map = hue_start + (hue_end - hue_start) * Hboost
    kC_map = kC_lo + (kC_hi - kC_lo) * Hboost
    theta = np.deg2rad(hue_deg_map)

    dA = kC_map * np.cos(theta) * fix_ms
    dB = kC_map * np.sin(theta) * fix_ms
    dL = (dL_hot_hi * Hboost) * fix_ms

    L2 = np.clip(L + dL, 0, 100)
    A2 = np.clip(A + dA, -128, 127)
    B2 = np.clip(B + dB, -128, 127)

    lab_out = np.stack([L2, A2, B2], axis=-1)
    lab_out[~tissue_mask] = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    rgb_out = np.clip(color.lab2rgb(lab_out), 0, 1)

    # ====== LOAD SLIDE MASK + CONTOURS ======
    imgPathMask = os.path.join(slide_mask_path, slide_name + '_mask.tif')
    with openslide.OpenSlide(imgPathMask) as mask_slide:
        nx, ny = mask_slide.dimensions
        thumb_mask = mask_slide.get_thumbnail((nx // down_scale, ny // down_scale))
        thumb_mask_grey = np.array(thumb_mask.convert('L'))

    gt_mask = (thumb_mask_grey > 0).astype(np.uint8)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert rgb_out [0,1] float → uint8 for OpenCV
    rgb_out_uint8 = (rgb_out * 255).astype(np.uint8)
    cv2.drawContours(rgb_out_uint8, gt_contours, -1, gt_color, 3)

    # ====== COLORBAR (true value→hue sweep) ======
    tissue_pixels = lab_thumb[tissue_mask]
    L0, A0, B0 = float(np.mean(tissue_pixels[...,0])), float(np.mean(tissue_pixels[...,1])), float(np.mean(tissue_pixels[...,2]))
    t = np.linspace(0, 1, 256); tboost = t**nonlin_pow
    hue_bar = hue_start + (hue_end - hue_start) * tboost
    kC_bar  = kC_lo + (kC_hi - kC_lo) * tboost
    theta_b = np.deg2rad(hue_bar)
    dA_bar = kC_bar * np.cos(theta_b); dB_bar = kC_bar * np.sin(theta_b); dL_bar = dL_hot_hi * tboost
    L_bar = np.clip(L0 + dL_bar, 0, 100); A_bar = np.clip(A0 + dA_bar, -128, 127); B_bar = np.clip(B0 + dB_bar, -128, 127)
    lab_bar = np.stack([L_bar, A_bar, B_bar], axis=-1)
    rgb_bar = np.clip(color.lab2rgb(lab_bar.reshape(1, -1, 3)), 0, 1).reshape(-1, 3)
    cmap_lab_sweep = mpl.colors.ListedColormap(rgb_bar, name="lab_value_sweep")
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_lab_sweep); mappable.set_array([])

    # ====== PLOT ======
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_out_uint8)
    ax.set_title('Tissue and Fixation Heatmap (tumors within green contours)')

    cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.04, shrink=0.75)
    cbar.set_label('Fixation reduction value (gamma stretch)')

    fig.savefig(f'./fixation_heatmap_prod/{slide_name}_{example_heatmap.split(".npy")[0]}.png', dpi=300, bbox_inches='tight')
