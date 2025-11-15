# %%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import openslide
import os.path as osp
import os
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom
from scipy.ndimage import label, sum as nd_sum
import matplotlib.ticker as mtick


# %%
def get_binary_mask(img_path, threshold=0.9, target_size=None):
    """
    Get binary mask with specific target size.
    target_size: tuple of (width, height) for the output mask
    """
    with Image.open(img_path) as img:
        gray = img.convert('L')
        gray_array = np.array(gray)/255.0
        binary_mask = gray_array < (1-threshold)
    
    if target_size is not None:
        # Calculate zoom factors to match target size exactly
        zoom_y = target_size[1] / binary_mask.shape[0]
        zoom_x = target_size[0] / binary_mask.shape[1]
        shrunk_mask = zoom(binary_mask, zoom=(zoom_y, zoom_x), order=0)
        
        # Double check dimensions
        if shrunk_mask.shape != (target_size[1], target_size[0]):
            shrunk_mask = shrunk_mask[:target_size[1], :target_size[0]]
    else:
        shrink_factor = 32 / 224  # shrink from 32 downsize to 224 downsize
        shrunk_mask = zoom(binary_mask, zoom=shrink_factor, order=0)

    return shrunk_mask


def calculate_iou(mask1, mask2):
    """Calculate IoU (Intersection over Union) between two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    union_sum = np.sum(union)
    if union_sum > 0:
        return np.sum(intersection) / union_sum
    return 0.0

def calculate_metrics(prediction, target):
    """Calculate precision, recall, and F1 score between prediction and target masks.
    
    Args:
        prediction: Binary mask of predictions
        target: Binary mask of ground truth
    
    Returns:
        tuple: (precision, recall, f1)
    """
    intersection = np.logical_and(prediction, target)
    
    pred_sum = np.sum(prediction)
    target_sum = np.sum(target)
    inter_sum = np.sum(intersection)
    
    precision = inter_sum / pred_sum if pred_sum > 0 else 0.0
    recall = inter_sum / target_sum if target_sum > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return precision, recall, f1


# %%
### open ground truth slide to compare
PATCH_SIZE = 224
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
BASE_TRUTH_DIR = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing_masks/'
CLAM_HEATMAP_PATH = '/users/PAS1575/chen8028/CLAM/heatmaps/production_results/cam16_test_gray_mask/tumor_tissue/'
FINETUNE_GAZE_05_S6_PATH = '/users/PAS1575/chen8028/CLAM/heatmaps/production_results/cam16_test_gray_mask_gaze_finetuned_0.5_S6/tumor_tissue/'
heatmap_ext = '_grascale.png'
image_list = [x.replace(heatmap_ext,'') for x in os.listdir(CLAM_HEATMAP_PATH) if 'grascale' in x]
print(f'Found {len(image_list)} images to process.')



# %% 
# loop through images to calculate IoU and other metrics

for image in image_list:
    output_file = f'./results_CLAM_vs_expanded_mask/{image}.csv'
    if osp.exists(output_file):
        print(f'Skipping {image}, results already exist.')
        continue

    res = []
    print(f'Processing image {image}...')

    # open baseline mask file
    truth_slide_path = osp.join(BASE_TRUTH_DIR, image + '_mask.tif')
    with openslide.open_slide(str(truth_slide_path)) as truth:
        thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / PATCH_SIZE, 
                    truth.dimensions[1] / PATCH_SIZE))
        slide_width, slide_height = thumbnail_truth.size
        
    thumbnail_truth = np.array(thumbnail_truth.convert("L"))
    if False:
        print(np.amin(thumbnail_truth), np.amax(thumbnail_truth), np.mean(thumbnail_truth))
        print(np.sum(thumbnail_truth))
        plt.imshow(thumbnail_truth, cmap='binary')
        print(thumbnail_truth.shape)    
    
    for THRESHOLD in THRESHOLDS:
        print(f'Threshold: {THRESHOLD}')

        ### get gaze_0.5_s6 finetuned heatmap mask
        img_path = osp.join(FINETUNE_GAZE_05_S6_PATH, image + heatmap_ext)
        binary_mask = get_binary_mask(img_path, threshold=THRESHOLD, target_size=(slide_width, slide_height))
        
        # Calculate all metrics using helper functions
        iou = calculate_iou(thumbnail_truth, binary_mask)
        precision, recall, f1 = calculate_metrics(binary_mask, thumbnail_truth)
        print(f'CLAM_PFMap_finetuned: precision {precision:0.3f}, recall {recall:0.3f}, f1-value {f1:0.3f}, IoU {iou:0.3f}')
        res.append([image, THRESHOLD, precision, recall, f1, iou, 'CLAM_PFMap_finetuned'])

        ### get CLAM heatmap mask
        img_path = osp.join(CLAM_HEATMAP_PATH, image + heatmap_ext)
        binary_mask = get_binary_mask(img_path, threshold=THRESHOLD, target_size=(slide_width, slide_height))
        
        # Calculate all metrics using helper functions
        iou = calculate_iou(thumbnail_truth, binary_mask)
        precision, recall, f1 = calculate_metrics(binary_mask, thumbnail_truth)
        print(f'CLAM_original       : precision {precision:0.3f}, recall {recall:0.3f}, f1-value {f1:0.3f}, IoU {iou:0.3f}')
        res.append([image, THRESHOLD, precision, recall, f1, iou, 'CLAM_original'])

        ### get expanded mask *****************************
        df = pd.read_csv(f'whole_slide_prediction/{image}.csv')
        mask = np.zeros((slide_height, slide_width), dtype=np.float32)

        # Fill the mask with the probability values from yhat at the given x and y coordinates
        for _, row in df.iterrows():
            x, y, yhat = int(row['x']), int(row['y']), row['yhat']
            mask[y, x] = yhat  # y is row index, x is column index            

        # Threshold the mask to create a binary mask (1 for tumor areas, 0 for others)
        binary_mask = (mask > THRESHOLD).astype(int)

        # Label connected components in the binary mask
        labeled_mask, num_features = label(binary_mask)
        
        # Calculate the size of each component
        component_sizes = nd_sum(binary_mask, labeled_mask, index=np.arange(1, num_features + 1))

        # Filter out small components
        size_threshold = 1  # Minimum size of regions to keep
        cleaned_binary_mask = binary_mask.copy()
        if component_sizes.size > 0:
            for i, size in enumerate(component_sizes, 1):
                if size < size_threshold:
                    cleaned_binary_mask[labeled_mask == i] = 0

        # Calculate all metrics using helper functions
        iou = calculate_iou(thumbnail_truth, cleaned_binary_mask)
        precision, recall, f1 = calculate_metrics(cleaned_binary_mask, thumbnail_truth)
        print(f'PFMap_supervised    : precision {precision:0.3f}, recall {recall:0.3f}, f1-value {f1:0.3f}, IoU {iou:0.3f}')
        res.append([image, THRESHOLD, precision, recall, f1, iou, 'PFMap_supervised'])

    # save results to csv
    df = pd.DataFrame(res, columns=['Image ID','Threshold','precision','recall','f1','IoU','Model'])
    df.to_csv(output_file, index=False)

print('done!')



# %%### combine all results into one csv
all_results = []
for image in image_list:
    input_file = f'./results_CLAM_vs_expanded_mask/{image}.csv'
    df = pd.read_csv(input_file)
    all_results.append(df)
combined_df = pd.concat(all_results, ignore_index=True)
combined_df



# %%
### make a box plot of the results -- IoU

fig, ax = plt.subplots(figsize=(8,4))
sns.boxplot(data=combined_df, x='Threshold', y='IoU', hue='Model', width=0.6)
plt.ylim((0.0,1))
plt.ylabel('IoU (with groundtruth)')
plt.xlabel('Threshold for tumor probability')
plt.title('Camelyon16 tumor test slides -- IoU at different thresholds', y=1.1)

# put legend on top
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          fancybox=True, shadow=False, ncol=3)

plt.savefig('./results_plots/CLAM_vs_supervised_IoU_box_plots.png', dpi=300, bbox_inches='tight')


# %%
### make a bar plot of the results -- precision or recall
metric = 'precision'   # 'precision' or 'recall' or 'f1'
df = combined_df.copy()

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(data=df, x='Threshold', y=metric, hue='Model', capsize=0.12, ax=ax)
plt.ylim((0,1))
plt.ylabel(metric.capitalize() + ' (relative to groundtruth)')
plt.xlabel('Threshold for tumor probability')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          fancybox=True, shadow=False, ncol=3)
ax.set_title(f'Camelyon16 tumor test slides -- {metric.capitalize()} at different thresholds', 
             y = 1.1)
plt.savefig(f'./results_plots/CLAM_vs_supervised_{metric}_box_plots.png', dpi=300, bbox_inches='tight')


# %%
