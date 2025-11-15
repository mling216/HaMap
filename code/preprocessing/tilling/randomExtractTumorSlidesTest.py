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
def cal_extract_allocation():

    ### get tumor testing slide names
    df_images = pd.read_csv('cam16_test_reference.csv')
    df_images = df_images[df_images['type']!='Normal']
    slides_to_test = df_images['image_id'].to_list()
    # print(slides_to_test)

    ### build a dataframe of all the tissue patches
    sampletotal_test = pd.DataFrame([])
    slide_info = {}

    for selection in slides_to_test:
        slide_file = SLIDE_PATH + selection +'.tif'
        with openslide.open_slide(slide_file) as slide:
            thumbnail = slide.get_thumbnail((slide.dimensions[0] / PATCH_SIZE, 
                        slide.dimensions[1] / PATCH_SIZE))
        # if selection[0]=='test_105':
        #     plt.imshow(thumbnail)
        #     thum = np.array(thumbnail)
        #     print(selection[0], thum.shape, np.amin(thum), np.amax(thum))
            
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
        
        # open baseline mask file
        truth_slide_path = BASE_TRUTH_DIR + selection + '_mask.tif'
        with openslide.open_slide(str(truth_slide_path)) as truth:
            thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / PATCH_SIZE, 
                        truth.dimensions[1] / PATCH_SIZE)) 
        """
        if selection[0]=='test_105':
            thum = np.array(thumbnail_truth)
            ### need to convert 3 channels (x,y,3) to gray (x,y)
            thum = cv2.cvtColor(thum, cv2.COLOR_BGR2GRAY)
            plt.imshow(thum)
            ### OR x255 to be in color space, like below
            # plt.imshow(thum*255)
            print(selection[0], thum.shape, np.amin(thum), np.amax(thum))
        """ 

        patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack())
        patches_y['is_tumor'] = patches_y[0] > 0
        patches_y.drop(0, axis=1, inplace=True)

        samples = pd.concat([patches, patches_y], axis=1)

        # remove non_tissue tiles
        samples = samples[samples.is_tissue == True] # remove patches with no tissue
        samples['tile_loc'] = list(samples.index)
        samples.reset_index(inplace=True, drop=True)

        # sampletotal_test = sampletotal_test.append(samples, ignore_index=True)    # will give deprecate warning
        sampletotal_test = pd.concat([sampletotal_test, samples], ignore_index=True)

    ### count number of patches of each case for train
    sampletotal_test_tumor = sampletotal_test[sampletotal_test['is_tumor'] == True]
    n_test_tumor_patches = len(sampletotal_test_tumor)
    sampletotal_test_normal = sampletotal_test[(sampletotal_test['is_tumor'] == False) &
                                                 (sampletotal_test['is_tissue'] == True)]
    n_test_normal_patches = len(sampletotal_test_normal)

    print('test tumor patches', n_test_tumor_patches, 'test normal patches', n_test_normal_patches)

    ### extract uniformlly across slides
    extract_allocation = []

    # train slides
    list_slides = list(sampletotal_test_tumor['slide_path'].unique())
    total_to_sample_test_tumor = 0
    total_to_sample_test_normal = 0
    for i, slide_file in enumerate(sorted(list_slides)):
        df_samples_tumor = sampletotal_test_tumor[sampletotal_test_tumor['slide_path'] == slide_file]
        df_samples_normal = sampletotal_test_normal[sampletotal_test_normal['slide_path'] == slide_file]
        # calculate the num tiles to sample for current slide
        if i == len(list_slides)-1:
            num_to_extract_tumor = NUM_SAMPLES - total_to_sample_test_tumor
            num_to_extract_normal = round(NUM_SAMPLES/2) - total_to_sample_test_normal
        else:
            num_to_extract_tumor = round(len(df_samples_tumor) * NUM_SAMPLES / n_test_tumor_patches)
            num_to_extract_normal = round(len(df_samples_normal) * (NUM_SAMPLES/2) / n_test_normal_patches)
        extract_allocation.append([slide_file, num_to_extract_tumor, num_to_extract_normal])
        total_to_sample_test_tumor += num_to_extract_tumor
        total_to_sample_test_normal += num_to_extract_normal
        print(osp.basename(slide_file), len(df_samples_tumor), num_to_extract_tumor, 
              len(df_samples_normal), num_to_extract_normal)

    # sort by decreasing tumor-tile-size
    extract_allocation = sorted(extract_allocation, key=lambda x: x[1], reverse=True)
    for ex_alloc in extract_allocation:
        print(ex_alloc)

    print(total_to_sample_test_tumor, total_to_sample_test_normal)
    return slide_info, extract_allocation


### convert annotation polygons to df
def convert_xml_df (file):
    parseXML = et.parse(file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
            Y_coord = float(coordinate.attrib.get('Y'))
            df_xml = pd.concat([df_xml, pd.DataFrame([[Name, Order, X_coord, Y_coord]], columns=dfcols)], ignore_index=True)
    return (df_xml)


### randomly crop a tile from within the tumor lesion box
def random_crop_tumor(slide, truth, bbox):
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
        
    rgb_mask = truth.read_region((x, y), 0, CROP_SIZE)
    rgb_mask = (cv2.cvtColor(np.array(rgb_mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)

    rgb_image = slide.read_region((x, y), 0, CROP_SIZE)
    rgb_array = np.array(rgb_image)

    return (rgb_array, rgb_mask, index)


### randomly crop a tile from within the tissue lesion box
def random_crop_tissue(slide, truth, thresh, bbox):
    dy, dx = CROP_SIZE
    x = np.random.randint(bbox[0], bbox[1] - dx + 1)
    y = np.random.randint(bbox[2], bbox[3] - dy + 1)
    index=[x, y]

    rgb_mask = truth.read_region((x, y), 0, CROP_SIZE)
    rgb_mask = (cv2.cvtColor(np.array(rgb_mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)

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
def random_extract(slide_info, extract_allocation, output_path_test_tumor, output_path_test_normal):

    for allocation in extract_allocation:
        slide_file = allocation[0]
        num_to_extract_tumor = allocation[1]
        num_to_extract_normal = allocation[2]
        output_path_tumor = output_path_test_tumor
        output_path_normal = output_path_test_normal
        slide_name = osp.basename(slide_file).split('.')[0]
        # print(slide_name)

        ### check # tiles already extracted and calculate the remaining #
        num_tumor_already = len(glob.glob(osp.join(output_path_tumor, '%s_*.png' % slide_name)))
        num_to_extract_tumor = num_to_extract_tumor - num_tumor_already
        num_normal_already = len(glob.glob(osp.join(output_path_normal, '%s_*.png' % slide_name)))
        num_to_extract_normal = num_to_extract_normal - num_normal_already
        print(slide_name, num_tumor_already, num_to_extract_tumor, num_normal_already, num_to_extract_normal)

        truth_slide_path = BASE_TRUTH_DIR + slide_name + '_mask.tif'
        Anno_pathxml = ANNOTATION_PATH + slide_name + '.xml'

        # open slide image
        with openslide.open_slide(slide_file) as slide:
            
            # open groundtruth mask and get tumor patches
            with openslide.open_slide(str(truth_slide_path)) as truth:

                # calculate 'rectangle' tumor bounding box based on xml polygons
                annotations = convert_xml_df(str(Anno_pathxml))
                x_values = list(annotations['X'].values)
                y_values = list(annotations['Y'].values)
                bbox_tumor = math.floor(np.min(x_values)), math.floor(np.max(x_values)), math.floor(np.min(y_values)), math.floor(np.max(y_values))

                # get tumor patches from tumor areas
                m=0
                while m in range(0, num_to_extract_tumor):
                    res = random_crop_tumor(slide, truth, bbox_tumor)
                    if (cv2.countNonZero(res[1]) > CROP_SIZE[0]*CROP_SIZE[1]*0.5) and (m <= num_to_extract_tumor):    
                        if (m) % 100 == 0: 
                            print(m)
                        ### random pick one from five choices: self, mirror, 90 rotate, 180 rotate, 270 rotate
                        img, choice = random_rotate(res[0])
                        imsave(output_path_tumor + '%s_%d_%d_%d.png' % (slide_name, res[2][0], res[2][1], choice), img)
                        m = m + 1

                """
                # get normal patches from tissue areas outside of tumor areas
                n=0
                while n in range(0, num_to_extract_normal):
                    res = random_crop_tissue(slide, truth, slide_info[slide_file][0], slide_info[slide_file][1])
                    if (cv2.countNonZero(res[1]) > CROP_SIZE[0]*CROP_SIZE[1]*0.2) and \
                            (cv2.countNonZero(res[2]) == 0) and \
                            (n <= num_to_extract_normal):    
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
NUM_SAMPLES = 50000
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing/images/'
BASE_TRUTH_DIR = '/fs/ess/PAS1575/Dataset/CAMELYON16/testing_masks/'
ANNOTATION_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/lesion_annotations_test/'


### run the extraction
def main():
    # create output path
    output_path_test_tumor = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMask/test/tumor/'
    os.makedirs(output_path_test_tumor, exist_ok=True)
    output_path_test_normal = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMask/test/normal/test_tumor/'
    # os.makedirs(output_path_test_normal, exist_ok=True)

    slide_info, extract_allocation = cal_extract_allocation()
    random_extract(slide_info, extract_allocation, output_path_test_tumor, output_path_test_normal)



# load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    main()
    

# %%
