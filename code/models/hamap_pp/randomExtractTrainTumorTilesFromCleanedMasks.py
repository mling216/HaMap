# %%
import numpy as np
import cv2
import openslide
import os
import random
from skimage.io import imsave
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import glob


def set_tile_allocation(df_info, seed):
    # Split the slide names into training and validation sets
    train_slides, val_slides = train_test_split(df_info, test_size=VAL_PROPORTION, random_state=seed)

    # Calculate total tumor size for training and validation sets
    total_tumor_size_train = train_slides['tumor_size'].sum()
    total_tumor_size_val = val_slides['tumor_size'].sum()

    # Calculate the number of tiles to allocate for train and validation sets
    num_train_tiles = int(NUM_SAMPLES * (1-VAL_PROPORTION))
    num_val_tiles = NUM_SAMPLES - num_train_tiles

    # Allocate tiles to each slide in the training set based on proportion of tumor size
    train_slides['tile_proportion'] = train_slides['tumor_size'] / total_tumor_size_train
    train_slides['n_tiles_to_extract'] = (train_slides['tile_proportion'] * num_train_tiles).round().astype(int)
    train_slides = train_slides.sort_values(by='n_tiles_to_extract', ascending=False)
    print(f"total train tiles: {sum(train_slides['n_tiles_to_extract'])}; tile allocation:")
    print(train_slides[['slide_name','n_tiles_to_extract']])

    # Allocate tiles to each slide in the validation set based on proportion of tumor size
    val_slides['tile_proportion'] = val_slides['tumor_size'] / total_tumor_size_val
    val_slides['n_tiles_to_extract'] = (val_slides['tile_proportion'] * num_val_tiles).round().astype(int)
    val_slides = val_slides.sort_values(by='n_tiles_to_extract', ascending=False)    
    print(f"total validation tiles: {sum(val_slides['n_tiles_to_extract'])}; tile allocation:")
    print(val_slides[['slide_name','n_tiles_to_extract']])

    # Generate the list with [slide_name, n_tiles_to_extract, train_or_val]
    tile_allocation = []

    # Add train slides
    for slide_name, n_tiles, thresh, mask_name in train_slides[['slide_name', 'n_tiles_to_extract', 'tissue_thresh', 'mask_name']].values:
        tile_allocation.append([slide_name, n_tiles, thresh, mask_name, 'train'])

    # Add validation slides
    for slide_name, n_tiles, thresh, mask_name in val_slides[['slide_name', 'n_tiles_to_extract', 'tissue_thresh', 'mask_name']].values:
        tile_allocation.append([slide_name, n_tiles, thresh, mask_name, 'val'])

    # return output list
    return tile_allocation


def set_scaled_tile_allocation(df_info, seed, bottom=100, reduction=10):
    # Split the slide names into training and validation sets
    train_slides, val_slides = train_test_split(df_info, test_size=VAL_PROPORTION, random_state=seed)

    # Calculate total tumor size for training and validation sets
    total_tumor_size_train = train_slides['tumor_size'].sum()
    total_tumor_size_val = val_slides['tumor_size'].sum()

    # Calculate the number of tiles to allocate for train and validation sets
    num_train_tiles = int(NUM_SAMPLES * (1 - VAL_PROPORTION))
    num_val_tiles = NUM_SAMPLES - num_train_tiles

    # Allocate tiles to each slide in the training set based on proportion of tumor size
    train_slides['tile_proportion'] = train_slides['tumor_size'] / total_tumor_size_train
    train_slides['n_tiles_to_extract'] = (train_slides['tile_proportion'] * num_train_tiles).round().astype(int)

    # Allocate tiles to each slide in the validation set based on proportion of tumor size
    val_slides['tile_proportion'] = val_slides['tumor_size'] / total_tumor_size_val
    val_slides['n_tiles_to_extract'] = (val_slides['tile_proportion'] * num_val_tiles).round().astype(int)

    # Scaling for train tiles
    min_train_tiles = bottom
    max_train_tiles = num_train_tiles // reduction

    train_slides['n_tiles_to_extract'] = (
        ((train_slides['n_tiles_to_extract'] - train_slides['n_tiles_to_extract'].min()) /
         (train_slides['n_tiles_to_extract'].max() - train_slides['n_tiles_to_extract'].min())) * 
        (max_train_tiles - min_train_tiles) + min_train_tiles
    ).round().astype(int)

    # Normalize to ensure the total number of train tiles is still num_train_tiles
    train_slides['n_tiles_to_extract'] = (
        train_slides['n_tiles_to_extract'] / train_slides['n_tiles_to_extract'].sum() * num_train_tiles
    ).round().astype(int)

    # Scaling for validation tiles
    min_val_tiles = bottom
    max_val_tiles = num_val_tiles // reduction

    val_slides['n_tiles_to_extract'] = (
        ((val_slides['n_tiles_to_extract'] - val_slides['n_tiles_to_extract'].min()) /
         (val_slides['n_tiles_to_extract'].max() - val_slides['n_tiles_to_extract'].min())) * 
        (max_val_tiles - min_val_tiles) + min_val_tiles
    ).round().astype(int)

    # Normalize to ensure the total number of val tiles is still num_val_tiles
    val_slides['n_tiles_to_extract'] = (
        val_slides['n_tiles_to_extract'] / val_slides['n_tiles_to_extract'].sum() * num_val_tiles
    ).round().astype(int)

    # Sort train and validation slides by 'n_tiles_to_extract' in descending order
    train_slides = train_slides.sort_values(by='n_tiles_to_extract', ascending=False)
    val_slides = val_slides.sort_values(by='n_tiles_to_extract', ascending=False)

    print(f"total train tiles: {sum(train_slides['n_tiles_to_extract'])}; tile allocation:")
    print(train_slides[['slide_name', 'n_tiles_to_extract']])

    print(f"total train tiles: {sum(val_slides['n_tiles_to_extract'])}; tile allocation:")
    print(val_slides[['slide_name', 'n_tiles_to_extract']])

    # Generate the list with [slide_name, n_tiles_to_extract, train_or_val]
    tile_allocation = []

    # Add train slides
    for slide_name, n_tiles, thresh, mask_name in train_slides[['slide_name', 'n_tiles_to_extract', 'tissue_thresh', 'mask_name']].values:
        tile_allocation.append([slide_name, n_tiles, thresh, mask_name, 'train'])

    # Add validation slides
    for slide_name, n_tiles, thresh, mask_name in val_slides[['slide_name', 'n_tiles_to_extract', 'tissue_thresh', 'mask_name']].values:
        tile_allocation.append([slide_name, n_tiles, thresh, mask_name, 'val'])

    # return output list
    return tile_allocation


### random pick one from five choices: self, mirror, 90 rotate, 180 rotate, 270 rotate
def random_rotate(rgb_array):
    choice = random.randint(0,4)    # include upper bound
    if choice==0:
        img = rgb_array
    elif choice==1:
        img = np.fliplr(rgb_array)
    elif choice==2:
        img = np.rot90(rgb_array, 1)
    elif choice==3:
        img = np.rot90(rgb_array, 2)
    elif choice==4:
        img = np.rot90(rgb_array, 3)
    
    return img, choice


def random_crop_tumor(slide, truth, thresh, bbox):
    dy, dx = CROP_SIZE
    bbox_x0, bbox_x1, bbox_y0, bbox_y1 = bbox
    
    if bbox_x0 >= bbox_x1 - dx + 1:
        x = bbox_x0
    else:
        x = np.random.randint(bbox_x0, bbox_x1 - dx + 1)
    
    if bbox_y0 >= bbox_y1 - dy + 1:
        y = bbox_y0
    else:
        y = np.random.randint(bbox_y0, bbox_y1 - dy + 1)
    
    index = [x, y]
        
    # Get gaze mask in the truth (mask) at the scaled-down location
    mask_x = (x - bbox_x0) // PATCH_SIZE
    mask_y = (y - bbox_y0) // PATCH_SIZE
    rgb_mask = truth[mask_y, mask_x] > 0

    # Extract the region from the slide
    rgb_image = slide.read_region((x, y), 0, CROP_SIZE)
    rgb_array = np.array(rgb_image)
    
    # Convert to HSV and threshold
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])

    return (rgb_array, rgb_binary, rgb_mask, index)


def prepare_bbox_and_save_tiles(slide_name, mask_name, threshold, num_to_extract, output_path):
    
    slide = openslide.OpenSlide(os.path.join(SLIDE_PATH, f'{slide_name}.tif'))
    
    # Load the filtered mask from the .npy file
    filtered_mask_cropped = np.load(os.path.join(MASK_PATH, mask_name))
    
    # Extract the original top-left coordinates (downscaled by 224) from the filename
    parts = mask_name.split('_')
    min_x = int(parts[-2])  # x coordinate from filename
    min_y = int(parts[-1].split('.')[0])  # y coordinate from filename

    # Convert min_x and min_y to original slide coordinates
    min_x_original = min_x * PATCH_SIZE
    min_y_original = min_y * PATCH_SIZE

    # Determine the bounding box within the cropped mask
    non_zero_coords = np.argwhere(filtered_mask_cropped > 0)
    if non_zero_coords.size == 0:
        raise ValueError("The mask does not contain any tumor region.")
    
    # Get the bounding box from the non-zero region
    bbox_min_y, bbox_min_x = non_zero_coords.min(axis=0)
    bbox_max_y, bbox_max_x = non_zero_coords.max(axis=0)

    # Convert bounding box coordinates to slide coordinates
    bbox_x0 = min_x_original + bbox_min_x * PATCH_SIZE
    bbox_x1 = min_x_original + (bbox_max_x + 1) * PATCH_SIZE
    bbox_y0 = min_y_original + bbox_min_y * PATCH_SIZE
    bbox_y1 = min_y_original + (bbox_max_y + 1) * PATCH_SIZE

    ### check num tiles already extracted and calculate the remaining #
    files_arealdy = glob.glob(os.path.join(output_path, f'{slide_name}_*.png'))
    num_extracted_already = len(files_arealdy)
    num_remaining_to_extract = num_to_extract - num_extracted_already
    # delete extra tiles
    if num_remaining_to_extract<0:
        num_extra = abs(num_remaining_to_extract)
        files_to_delete = random.sample(files_arealdy, num_extra)
        print('start to delete extra files')
        for file_path in files_to_delete:
            os.remove(file_path)
        print(f"Deleted extra files: {num_extra}")
        num_remaining_to_extract = 0
    print(f'{slide_name}, total to extract {num_to_extract}, already extracted {num_extracted_already}, left to extract {num_remaining_to_extract}') 

    # Generate and save multiple tumor tiles
    m = 0  # Counter for the number of saved tiles
    while m < num_remaining_to_extract:
        res = random_crop_tumor(slide, filtered_mask_cropped, threshold, (bbox_x0, bbox_x1, bbox_y0, bbox_y1))
        
        # Check if the tile meets the conditions
        if (cv2.countNonZero(res[1]) > CROP_SIZE[0] * CROP_SIZE[1] * 0.25) and res[2] == True:
            if m % 100 == 0 and m>0:
                print(f"Saved {m} tiles.")
            
            # Randomly rotate and/or mirror the image
            img, choice = random_rotate(res[0])
            
            # Save the image
            output_filename = os.path.join(output_path, f'{slide_name}_{res[3][0]}_{res[3][1]}_{choice}.png')
            imsave(output_filename, img)
            m += 1


def get_threshold_from_string(thresh_string):
    # Regular expression to extract the numeric values
    matches = re.findall(r'array\(\[([0-9,\s]+)\]', thresh_string)

    # Convert the extracted strings into lists of integers and then into NumPy arrays
    thresh = [np.array(list(map(int, match.split(','))), dtype=np.uint8) for match in matches]

    return thresh


# Usage Example
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training/tumor'
MASK_PATH = 'converted_masks/'

PATCH_SIZE = 224
CROP_SIZE = (PATCH_SIZE, PATCH_SIZE)
NUM_SAMPLES = 100000
VAL_PROPORTION = 0.2

SLIDES_TUMOR_SIZE_FILE = 'train_tumor_slides_tumor_size.csv'
SLIDES_TISSUE_THRESHOLD_FILE = 'train_tumor_slides_tissue_thresholds.csv'

# read pre-obtained slide-level data and create an info dataframe
df_tumor_size = pd.read_csv(SLIDES_TUMOR_SIZE_FILE)
df_tissue_thresh = pd.read_csv(SLIDES_TISSUE_THRESHOLD_FILE)
df_info = pd.merge(df_tumor_size, df_tissue_thresh, on='slide_name')

mask_files = os.listdir(MASK_PATH)
df_mask = pd.DataFrame(mask_files, columns=['mask_name'])
df_mask['slide_name'] = df_mask['mask_name'].apply(lambda x: '_'.join(x.split('_')[:2]))
df_info = pd.merge(df_info, df_mask, on='slide_name')

df_info.sort_values('slide_name', ascending=False, inplace=True)

# get tile extraction allocation data
seed = 26
is_scaled = True   # whether to scale on top of size-weighted allocation

if is_scaled:
    tile_allocation = set_scaled_tile_allocation(df_info, seed, bottom=100, reduction=10)
else:
    tile_allocation = set_tile_allocation(df_info, seed)


# %%
# create output path
if is_scaled:
    output_path_train_tumor = f'/fs/scratch/PAS1575/Pathology/CAMELYON16/expandedIndividualMask/train/tumor_scaled_s{seed}'
    output_path_val_tumor = f'/fs/scratch/PAS1575/Pathology/CAMELYON16/expandedIndividualMask/validation/tumor_scaled_s{seed}'
else:
    output_path_train_tumor = f'/fs/scratch/PAS1575/Pathology/CAMELYON16/expandedIndividualMask/train/tumor_s{seed}'
    output_path_val_tumor = f'/fs/scratch/PAS1575/Pathology/CAMELYON16/expandedIndividualMask/validation/tumor_s{seed}'

os.makedirs(output_path_train_tumor, exist_ok=True)
os.makedirs(output_path_val_tumor, exist_ok=True)

for slide_name, n_tiles, thresh, mask_name, split in tile_allocation:
    # if slide_name!='tumor_001' and slide_name!='tumor_005':
    #     continue
    
    threshold = get_threshold_from_string(thresh)
    print(f"{split}, {slide_name}, {n_tiles}, threshold: {threshold}, mask file: {mask_name}")

    # Generate and save tiles
    if split=='train':
        output_path = output_path_train_tumor
    elif split=='val':
        output_path = output_path_val_tumor
    else:
        print('WRONG folder !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', split)

    prepare_bbox_and_save_tiles(slide_name, mask_name, threshold, n_tiles, output_path)

print('extraction done!')


# %%
