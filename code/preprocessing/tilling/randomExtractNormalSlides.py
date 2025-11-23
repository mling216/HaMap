# from scipy.misc import imsave # need to be <=1.2.0
from imageio import imwrite as imsave
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from skimage.filters import threshold_otsu
import glob
import cv2
import xml.etree.ElementTree as et
from collections import Counter
import math
import random
from sklearn.model_selection import train_test_split
import argparse as ap


### calculate how many tiles to extract from each slide
def cal_extract_allocation(seed):

    ### get normal training slide names    
    slides_all = glob.glob(osp.join(SLIDE_PATH, '*.tif'))
    slides_all = [x for x in slides_all if osp.basename(x)!='normal_039.tif']
    
    ### use 80% slides as train, and the other 20% as val
    train_slides, val_slides = train_test_split(slides_all, test_size=0.2, random_state=seed)

    # save selections in a list
    slides_to_test = []
    for slide_file in train_slides:
        slides_to_test.append([slide_file, 'train'])
    for slide_file in val_slides:
        slides_to_test.append([slide_file, 'val'])

    ### build a dataframe of all the tissue patches
    sampletotal_train = pd.DataFrame([])
    sampletotal_val = pd.DataFrame([])
    slide_info = {}

    for selection in slides_to_test:
        slide_file = selection[0]
        with openslide.open_slide(slide_file) as slide:
            thumbnail = slide.get_thumbnail((slide.dimensions[0] / PATCH_SIZE, 
                        slide.dimensions[1] / PATCH_SIZE))
            
        # get tissue area tiles
        thum = np.array(thumbnail)
        hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
        # be min value for v can be changed later
        minhsv = np.array([hthresh, sthresh, 70], np.uint8)
        maxhsv = np.array([180, 255, vthresh], np.uint8)
        thresh = [minhsv, maxhsv]

        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
        binary = rgbbinary / 255
        binary = binary.astype(np.uint8)

        ### find bounds from a mask -- more efficient
        ymax, xmax = np.max(np.where(binary>0), 1)  # note x and y are reverse
        ymin, xmin = np.min(np.where(binary>0), 1)
        bbox_tissue = [xmin*PATCH_SIZE, xmax*PATCH_SIZE, ymin*PATCH_SIZE, ymax*PATCH_SIZE]

        # save thresh and bbox for each slide for later use
        slide_info[slide_file] = [thresh, bbox_tissue]

        patches = pd.DataFrame(pd.DataFrame(binary).stack())
        patches['is_tissue'] = patches[0]       # not ~patches[0]
        patches.drop(0, axis=1, inplace=True)   # drop the True/False column
        patches['slide_path'] = slide_file
        
        samples = patches
        # remove non_tissue tiles
        samples = samples[samples.is_tissue == True] # remove patches with no tissue
        samples['tile_loc'] = list(samples.index)
        samples.reset_index(inplace=True, drop=True)

        if selection[1] == 'train':
            sampletotal_train = pd.concat([sampletotal_train, samples], ignore_index=True)
        elif selection[1] == 'val':
            sampletotal_val = pd.concat([sampletotal_val, samples], ignore_index=True)

    ### count number of patches of each case for train
    sampletotal_train_normal = sampletotal_train[(sampletotal_train['is_tissue'] == True)]
    n_train_normal_patches = len(sampletotal_train_normal)

    sampletotal_val_normal = sampletotal_val[(sampletotal_val['is_tissue'] == True)]
    n_val_normal_patches = len(sampletotal_val_normal)

    print('train normal patches', n_train_normal_patches, 'val normal patches', n_val_normal_patches)

    ### extract uniformlly across slides
    extract_allocation = []

    # train slides
    list_slides = list(sampletotal_train_normal['slide_path'].unique())
    total_to_sample_train_normal = 0
    for i, slide_file in enumerate(sorted(list_slides)):
        df_samples_normal = sampletotal_train_normal[sampletotal_train_normal['slide_path'] == slide_file]
        # calculate the num tiles to sample for current slide
        if i == len(list_slides)-1:
            num_to_extract_normal = round(NUM_SAMPLES*(1-VAL_PROPORTION)) - total_to_sample_train_normal
        else:
            num_to_extract_normal = round(len(df_samples_normal) * NUM_SAMPLES*(1-VAL_PROPORTION) / n_train_normal_patches)
        extract_allocation.append([slide_file, num_to_extract_normal, 'train'])
        total_to_sample_train_normal += num_to_extract_normal
        print(osp.basename(slide_file), len(df_samples_normal), num_to_extract_normal)
    print()

    # val slides
    list_slides = list(sampletotal_val_normal['slide_path'].unique())
    total_to_sample_val_normal = 0
    for i, slide_file in enumerate(sorted(list_slides)):
        df_samples_normal = sampletotal_val_normal[sampletotal_val_normal['slide_path'] == slide_file]
        # calculate the num tiles to sample for current slide
        if i == len(list_slides)-1:
            num_to_extract_normal = round(NUM_SAMPLES*VAL_PROPORTION) - total_to_sample_val_normal
        else:
            num_to_extract_normal = round(len(df_samples_normal) * (NUM_SAMPLES*VAL_PROPORTION) / n_val_normal_patches)
        extract_allocation.append([slide_file, num_to_extract_normal, 'val'])
        total_to_sample_val_normal += num_to_extract_normal
        print(osp.basename(slide_file), len(df_samples_normal), num_to_extract_normal)

    print(total_to_sample_train_normal, total_to_sample_val_normal)
    return slide_info, extract_allocation


### randomly crop a tile from within the tissue lesion box
def random_crop_tissue(slide, thresh, bbox):
    dy, dx = CROP_SIZE
    x = np.random.randint(bbox[0], bbox[1] - dx + 1)
    y = np.random.randint(bbox[2], bbox[3] - dy + 1)
    index=[x, y]

    rgb_image = slide.read_region((x, y), 0, CROP_SIZE)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])

    return (rgb_array, rgb_binary, index)


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


### extracting tiles randomly within both lesion and tissue regions
def random_extract(slide_info, extract_allocation, output_path_train_normal, output_path_val_normal):

    for allocation in extract_allocation:
        slide_file = allocation[0]
        num_to_extract_normal = allocation[1]
        if allocation[2]=='train':
            output_path_normal = output_path_train_normal
        elif allocation[2]=='val':
            output_path_normal = output_path_val_normal
        slide_name = osp.basename(slide_file).split('.')[0]
        print(slide_name)

        # open slide image
        with openslide.open_slide(slide_file) as slide:
            # get normal patches from tissue areas outside of tumor areas
            n=0
            while n in range(0, num_to_extract_normal):
                res = random_crop_tissue(slide, slide_info[slide_file][0], slide_info[slide_file][1])
                if (cv2.countNonZero(res[1]) > CROP_SIZE[0]*CROP_SIZE[1]*0.2) and (n <= num_to_extract_normal):    
                    if (n) % 100 == 0: 
                        print(n)
                    ### random pick one from five choices: self, mirror, 90 rotate, 180 rotate, 270 rotate
                    img, choice = random_rotate(res[0])
                    imsave(output_path_normal + '%s_%d_%d_%d.png' % (slide_name, res[2][0], res[2][1], choice), img)
                    n = n + 1



### global variables
PATCH_SIZE = 224
CROP_SIZE = (PATCH_SIZE, PATCH_SIZE)
NUM_SAMPLES = 100000
VAL_PROPORTION = 0.2
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training/normal/'


### run the extraction
def main(seed):
    # create output path
    output_path_train = '/fs/scratch/PAS1575/Pathology/CAMELYON16/extracted_tiles/train/normal/trainset_s%s/' % seed
    os.makedirs(output_path_train, exist_ok=True)
    output_path_val = '/fs/scratch/PAS1575/Pathology/CAMELYON16/extracted_tiles/validation/normal/trainset_s%s/' % seed
    os.makedirs(output_path_val, exist_ok=True)

    slide_info, extract_allocation = cal_extract_allocation(seed)
    random_extract(slide_info, extract_allocation, output_path_train, output_path_val)



# load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get dataset seed')
    parser.add_argument('-s', metavar='--seed', type=str, action='store', 
                        dest='seed', required=True, 
                        help='random seed for tile split')
    
    # Gather the provided arguments as an array.
    args = parser.parse_args()
    seed = vars(args)['seed']
    if not seed.isnumeric():
        print('seed %s is not numeric!' % seed)
    else:
        print('seed:', seed)
        main(int(seed))


# %%
