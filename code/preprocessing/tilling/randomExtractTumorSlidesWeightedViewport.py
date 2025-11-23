# %%
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
from PIL import Image
import xml.etree.ElementTree as et
from collections import Counter
import math
import random
from sklearn.model_selection import train_test_split
import argparse as ap


### calculate how many tiles to extract from each slide
def cal_extract_allocation(seed, weight_threshold):

    ### get train tumor slides
    df_slides = pd.read_csv(TRAIN_SLIDES_PARTICIPANT)
    df_slides = df_slides[df_slides['diagnosis']!='Benign']
    
    ##################### tumor_057 has errors: 
    df_slides = df_slides[df_slides['imageName']!='tumor_057']
    #####################

    masks = []
    for i, row in df_slides.iterrows():
        masks.append(['_'.join([row['imageID'],row['pID'],weight_threshold])+'.npy', row['imageName']])

    ### use 80% slides as train, and the other 20% as val
    train_slides, val_slides = train_test_split(masks, test_size=0.2, random_state=seed)

    # save selections in a list
    slides_to_test = []
    for mask, imageName in train_slides:
        slides_to_test.append([mask, imageName, 'train'])
    for mask, imageName in val_slides:
        slides_to_test.append([mask, imageName, 'val'])
    # print(slides_to_test)

    ### build a dataframe of all the tissue patches
    sampletotal_train = pd.DataFrame([])
    sampletotal_val = pd.DataFrame([])
    slide_info = {}

    for selection in slides_to_test:
        ############# diagnosis on a specific slide ############
        # if selection[1]!='test_001':
        #     continue
        ############# diagnosis on a specific slide ############

        # print(selection[0])

        slide_file = SLIDE_PATH + selection[1] +'.tif'
        with openslide.open_slide(slide_file) as slide:
            thumbnail = slide.get_thumbnail((slide.dimensions[0] / PATCH_SIZE, 
                        slide.dimensions[1] / PATCH_SIZE))
            # plt.imshow(thumbnail)
            # print(np.array(thumbnail).shape)

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

        patches = pd.DataFrame(pd.DataFrame(binary).stack())
        patches['is_tissue'] = patches[0]       # not ~patches[0]
        patches.drop(0, axis=1, inplace=True)   # drop the True/False column
        patches['slide_path'] = slide_file
        
        # open mask file and convert to patch_size
        mask_file = osp.join(GAZE_MASK_PATH, selection[0])
        mask = np.load(mask_file)
        
        ############# diagnosis on the gaze mask ############
        # print('%s (%s)' % (selection[0], selection[1]), 'number of 32 tiles is', mask.shape[0] * mask.shape[1])
        # print('number of tumor tiles at threshold %s is %d' % (weight_threshold, np.sum(mask)))
        # p = np.sum(mask)/(mask.shape[0] * mask.shape[1])
        # print('tumor area proportion is %0.2f' % p)
        # plt.imshow(mask)
        ############# diagnosis on the gaze mask ############
        
        new_h, new_w = mask.shape[0]/(PATCH_SIZE/32), mask.shape[1]/(PATCH_SIZE/32)

        # convert to Image image and resize
        mask_img = Image.fromarray(mask)
        mask_new = mask_img.resize((int(new_w), int(new_h)))       

        ### find bounds from a mask -- more efficient
        mask_new_array = np.array(mask_new)
        
        # skip if tumor region is too small and result in 0 cell
        if np.max(mask_new_array)==False:
            continue

        ymax, xmax = np.max(np.where(mask_new_array>0), 1)  # note x and y are reverse
        ymin, xmin = np.min(np.where(mask_new_array>0), 1)
        bbox_gaze = [xmin*PATCH_SIZE, xmax*PATCH_SIZE, ymin*PATCH_SIZE, ymax*PATCH_SIZE]

        # save thresh and bbox for each slide for later use
        slide_info[slide_file] = [thresh, bbox_tissue, bbox_gaze, mask_new_array]
        # print(mask_new_array.shape)

        patches_y = pd.DataFrame(pd.DataFrame(mask_new_array).stack())
        patches_y['is_tumor'] = patches_y[0] > 0
        patches_y.drop(0, axis=1, inplace=True)

        samples = pd.concat([patches, patches_y], axis=1)

        # remove non_tissue tiles
        samples = samples[samples['is_tissue'] == True] # remove patches with no tissue
        samples['tile_loc'] = list(samples.index)
        samples.reset_index(inplace=True, drop=True)

        if selection[2] == 'train':
            sampletotal_train = pd.concat([sampletotal_train, samples], ignore_index=True)
        elif selection[2] == 'val':
            sampletotal_val = pd.concat([sampletotal_val, samples], ignore_index=True)
    
    ############# diagnosis on a specific slide ############
    # return [], []
    ############# diagnosis on a specific slide ############

    ### count number of patches of each case for train
    sampletotal_train_tumor = sampletotal_train[sampletotal_train['is_tumor'] == True]
    n_train_tumor_patches = len(sampletotal_train_tumor)
    sampletotal_train_normal = sampletotal_train[(sampletotal_train['is_tumor'] == False) &
                                                 (sampletotal_train['is_tissue'] == True)]
    n_train_normal_patches = len(sampletotal_train_normal)

    sampletotal_val_tumor = sampletotal_val[sampletotal_val['is_tumor'] == True]
    n_val_tumor_patches = len(sampletotal_val_tumor)
    sampletotal_val_normal = sampletotal_val[(sampletotal_val['is_tumor'] == False) &
                                             (sampletotal_val['is_tissue'] == True)]
    n_val_normal_patches = len(sampletotal_val_normal)

    print('train tumor patches', n_train_tumor_patches, 'train normal patches', n_train_normal_patches)
    print('val tumor patches', n_val_tumor_patches, 'val normal patches', n_val_normal_patches)

    ### extract uniformlly across slides
    extract_allocation = []

    # train slides
    list_slides = list(sampletotal_train_tumor['slide_path'].unique())
    total_to_sample_train_tumor = 0
    total_to_sample_train_normal = 0
    for i, slide_file in enumerate(sorted(list_slides)):
        df_samples_tumor = sampletotal_train_tumor[sampletotal_train_tumor['slide_path'] == slide_file]
        df_samples_normal = sampletotal_train_normal[sampletotal_train_normal['slide_path'] == slide_file]
        # calculate the num tiles to sample for current slide
        if i == len(list_slides)-1:
            num_to_extract_tumor = max(0, round(NUM_SAMPLES*(1-VAL_PROPORTION)) - total_to_sample_train_tumor)  # avoid <0
            if len(df_samples_tumor) == 0:    # avoid last one > available
                num_to_extract_tumor = 0
            num_to_extract_normal = max(0, round(NUM_SAMPLES*((1-VAL_PROPORTION)/2)) - total_to_sample_train_normal)    # avoid <0
            if len(df_samples_normal) == 0:     # when 0 tiles, no extraction
                num_to_extract_normal = 0
        else:
            num_to_extract_tumor = round(len(df_samples_tumor) * NUM_SAMPLES*(1-VAL_PROPORTION) / n_train_tumor_patches)
            num_to_extract_normal = round(len(df_samples_normal) * NUM_SAMPLES*((1-VAL_PROPORTION)/2) / n_train_normal_patches)
        extract_allocation.append([slide_file, num_to_extract_tumor, num_to_extract_normal, 'train'])
        total_to_sample_train_tumor += num_to_extract_tumor
        total_to_sample_train_normal += num_to_extract_normal
        print(osp.basename(slide_file), len(df_samples_tumor), num_to_extract_tumor, 
              len(df_samples_normal), num_to_extract_normal)
    print()

    # val slides
    list_slides = list(sampletotal_val_tumor['slide_path'].unique())
    total_to_sample_val_tumor = 0
    total_to_sample_val_normal = 0
    for i, slide_file in enumerate(sorted(list_slides)):
        df_samples_tumor = sampletotal_val_tumor[sampletotal_val_tumor['slide_path'] == slide_file]
        df_samples_normal = sampletotal_val_normal[sampletotal_val_normal['slide_path'] == slide_file]
        # calculate the num tiles to sample for current slide
        if i == len(list_slides)-1:
            num_to_extract_tumor = max(0, round(NUM_SAMPLES*VAL_PROPORTION) - total_to_sample_val_tumor)    # avoid <0
            if len(df_samples_tumor) == 0:    # avoid last one > available
                num_to_extract_tumor = 0
            num_to_extract_normal = max(0, round(NUM_SAMPLES*VAL_PROPORTION/2) - total_to_sample_val_normal)    # avoid <0
            if len(df_samples_normal) == 0:     # when 0 tiles, no extraction
                num_to_extract_normal = 0
        else:
            num_to_extract_tumor = round(len(df_samples_tumor) * NUM_SAMPLES*VAL_PROPORTION / n_val_tumor_patches)
            num_to_extract_normal = round(len(df_samples_normal) * (NUM_SAMPLES*VAL_PROPORTION/2) / n_val_normal_patches)
        extract_allocation.append([slide_file, num_to_extract_tumor, num_to_extract_normal, 'val'])
        total_to_sample_val_tumor += num_to_extract_tumor
        total_to_sample_val_normal += num_to_extract_normal
        print(osp.basename(slide_file), len(df_samples_tumor), num_to_extract_tumor,
              len(df_samples_normal), num_to_extract_normal)

    # sort by decreasing tumor-tile-size
    extract_allocation = sorted(extract_allocation, key=lambda x: x[1], reverse=True)
    for ex_alloc in extract_allocation:
        print(ex_alloc)

    print(total_to_sample_train_tumor, total_to_sample_train_normal, total_to_sample_val_tumor, total_to_sample_val_normal)
    return slide_info, extract_allocation



### randomly crop a tile from within the eyetrack gaze box
def random_crop_tumor(slide, truth, thresh, bbox):
    dy, dx = CROP_SIZE
    if bbox[0] >= bbox[1] - dx + 1:
        x = bbox[0]
    else:
        x = np.random.randint(bbox[0], bbox[1] - dx + 1)
    if bbox[2] > bbox[3] - dy + 1:
        y = bbox[2]
    else:
        y = np.random.randint(bbox[2], bbox[3] - dy + 1)
    index=[x, y]
        
    # get gaze mask (y is 0th index)
    rgb_mask = truth[round(y/PATCH_SIZE),round(x/PATCH_SIZE)] > 0

    rgb_image = slide.read_region((x, y), 0, CROP_SIZE)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])

    return (rgb_array, rgb_binary, rgb_mask, index)



### randomly crop a tile from within the tissue lesion box
def random_crop_tissue(slide, truth, thresh, bbox):
    dy, dx = CROP_SIZE
    x = np.random.randint(bbox[0], bbox[1] - dx + 1)
    y = np.random.randint(bbox[2], bbox[3] - dy + 1)
    index=[x, y]

    # get gaze mask (y is 0th index)
    rgb_mask = truth[round(y/PATCH_SIZE),round(x/PATCH_SIZE)] > 0

    rgb_image = slide.read_region((x, y), 0, CROP_SIZE)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])

    return (rgb_array, rgb_binary, rgb_mask, index)



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
def random_extract(slide_info, extract_allocation, output_path_train_tumor, 
                   output_path_train_normal, output_path_val_tumor, output_path_val_normal):

    for allocation in extract_allocation:
        slide_file = allocation[0]
        num_to_extract_tumor = allocation[1]
        num_to_extract_normal = allocation[2]
        if allocation[3]=='train':
            output_path_tumor = output_path_train_tumor
            output_path_normal = output_path_train_normal
        elif allocation[3]=='val':
            output_path_tumor = output_path_val_tumor
            output_path_normal = output_path_val_normal
        slide_name = osp.basename(slide_file).split('.')[0]
        # print(slide_name)

        ### check # tiles already extracted and calculate the remaining #
        num_tumor_already = len(glob.glob(osp.join(output_path_tumor, '%s_*.png' % slide_name)))
        num_to_extract_tumor = num_to_extract_tumor - num_tumor_already
        num_normal_already = len(glob.glob(osp.join(output_path_normal, '%s_*.png' % slide_name)))
        num_to_extract_normal = num_to_extract_normal - num_normal_already
        print(slide_name, num_tumor_already, num_to_extract_tumor, num_normal_already, num_to_extract_normal)
        
        
        # open slide image
        with openslide.open_slide(slide_file) as slide:
            # slide_info = [thresh, bbox_tissue, bbox_gaze, mask_new_array]
            thresh = slide_info[slide_file][0]
            bbox_tissue = slide_info[slide_file][1]
            bbox_gaze = slide_info[slide_file][2]
            gaze_mask = slide_info[slide_file][3]

            # get 'tumor' patches from eyetrack gaze areas  
            m=0
            while m in range(0, num_to_extract_tumor):
                res = random_crop_tumor(slide, gaze_mask, thresh, bbox_gaze)
                # res <- (rgb_array, rgb_binary, rgb_mask, index)
                if (cv2.countNonZero(res[1]) > CROP_SIZE[0]*CROP_SIZE[1]*0.2) \
                    and res[2]==True and (m <= num_to_extract_tumor):    
                    if (m) % 100 == 0: 
                        print(m)
                    ### random pick one from five choices: self, mirror, 90 rotate, 180 rotate, 270 rotate
                    img, choice = random_rotate(res[0])
                    imsave(output_path_tumor + '%s_%d_%d_%d.png' % (slide_name, res[3][0], res[3][1], choice), img)
                    m = m + 1

            """
            # get normal patches from tissue areas outside of tumor areas
            n=0
            while n in range(0, num_to_extract_normal):
                res = random_crop_tissue(slide, gaze_mask, thresh, bbox_tissue)
                if (cv2.countNonZero(res[1]) > CROP_SIZE[0]*CROP_SIZE[1]*0.2) \
                        and res[2]==False and (n <= num_to_extract_normal):    
                    if (n) % 100 == 0: 
                        print(n)
                    ### random pick one from five choices: self, mirror, 90 rotate, 180 rotate, 270 rotate
                    img, choice = random_rotate(res[0])
                    imsave(output_path_normal + '%s_%d_%d_%d.png' % (slide_name, res[3][0], res[3][1], choice), img)
                    n = n + 1
            """



### global variables
PATCH_SIZE = 224
CROP_SIZE = (PATCH_SIZE, PATCH_SIZE)
NUM_SAMPLES = 100000
VAL_PROPORTION = 0.2
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training/all/'
GAZE_MASK_PATH = '/fs/ess/PAS1575/Dataset/new_data/fixation_masks/weighted_viewport/'
TRAIN_SLIDES_PARTICIPANT = './experimentData/trainSlidesCorrectDaignosis.csv'


### run the extraction
def main(seed, weight_threshold):
    # create output path
    output_path_train_tumor = '/fs/scratch/PAS1575/Pathology/CAMELYON16/extracted_tiles/train/tumor/w_viewport_%s_s%d/' % (weight_threshold, seed)
    os.makedirs(output_path_train_tumor, exist_ok=True)
    output_path_train_normal = '/fs/scratch/PAS1575/Pathology/CAMELYON16/extracted_tiles/train/normal/w_viewport_%s_s%d/' % (weight_threshold, seed)
    # os.makedirs(output_path_train_normal, exist_ok=True)

    output_path_val_tumor = '/fs/scratch/PAS1575/Pathology/CAMELYON16/extracted_tiles/validation/tumor/w_viewport_%s_s%d/' % (weight_threshold, seed)
    os.makedirs(output_path_val_tumor, exist_ok=True)
    output_path_val_normal = '/fs/scratch/PAS1575/Pathology/CAMELYON16/extracted_tiles/validation/normal/w_viewport_%s_s%d/' % (weight_threshold, seed)
    # os.makedirs(output_path_val_normal, exist_ok=True)

    slide_info, extract_allocation = cal_extract_allocation(seed, weight_threshold)
    random_extract(slide_info, extract_allocation, output_path_train_tumor, output_path_train_normal, 
                   output_path_val_tumor, output_path_val_normal)



############# for diagnosis ############
# main(6, '0.1')
############# for diagnosis ############


# %% 
### load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get dataset seed and eyetrack threshold')
    parser.add_argument('-s', metavar='--seed', type=str, action='store', 
                        dest='seed', required=True, 
                        help='random seed for tile split')
    parser.add_argument('-t', metavar='--threshold', type=str, action='store', 
                        dest='threshold', required=True, 
                        help='threshold for eyetrack extent')
    threshold_set = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']

    # Gather the provided arguments as an array.
    args = parser.parse_args()
    seed = vars(args)['seed']
    threshold = vars(args)['threshold']
    if not seed.isnumeric():
        print('seed %s is not numeric!' % seed)
    elif threshold not in threshold_set:
        print('threshold should be one of these:\n', threshold_set)
    else:
        print('seed:', seed)
        print('threshold:',threshold)
        main(int(seed), threshold)



# %%
