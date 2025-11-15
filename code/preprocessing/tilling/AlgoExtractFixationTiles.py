# %%
# from scipy.misc import imsave # need to be <=1.2.0
from imageio import imwrite as imsave
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from skimage.filters import threshold_otsu
import glob
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse as ap
import seaborn as sns



def calculateFixationTiles():
    ### get train tumor slides
    df_slides = pd.read_csv(TRAIN_SLIDES_PARTICIPANT)
    df_slides = df_slides[df_slides['diagnosis']!='Benign']
    mask_names = df_slides['maskName'].to_list()

    # save selections in a list
    slides_to_test = []
    for mask in mask_names:
        slides_to_test.append([mask, df_slides.loc[df_slides['maskName']==mask, 'imageName'].values[0]])

    ### calculate overlaps
    overlaps = []
    sample_fixation = pd.DataFrame([])
    tissue_thresh = []

    for selection in slides_to_test:
        ############# diagnosis on a specific slide ############
        # if selection[1]!='tumor_001' and selection[1]!='tumor_002':
        #     continue
        ############# diagnosis on a specific slide ############

        print(selection[0])

        ###
        ### get tissue tiles
        ###
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
        
        ## record the tissue thresholds
        tissue_thresh.append([selection[1], ','.join([str(x) for x in minhsv]), 
                              ','.join([str(x) for x in maxhsv])])

        rgbbinary = cv2.inRange(hsv_image, minhsv, maxhsv)
        binary = rgbbinary / 255
        binary = binary.astype(np.uint8)

        ### find bounds from a mask -- more efficient
        ymax, xmax = np.max(np.where(binary>0), 1)  # note x and y are reverse
        ymin, xmin = np.min(np.where(binary>0), 1)
        # bbox_tissue = [xmin*PATCH_SIZE, xmax*PATCH_SIZE, ymin*PATCH_SIZE, ymax*PATCH_SIZE]

        patches = pd.DataFrame(pd.DataFrame(binary).stack())
        patches['is_tissue'] = patches[0]       # not ~patches[0]
        patches.drop(0, axis=1, inplace=True)   # drop the True/False column
        patches['slide_path'] = slide_file


        ###
        ### get ground truth info
        ###
        slide_file = SLIDE_TRUTH + selection[1] +'_mask.tif'
        with openslide.open_slide(slide_file) as slide:
            thumbnail = slide.get_thumbnail((slide.dimensions[0] / PATCH_SIZE, 
                        slide.dimensions[1] / PATCH_SIZE))

            thumbnail_grey = np.array(thumbnail.convert('L'))/255.0
            thumbnail_grey = thumbnail_grey.astype(int)
            num_tumor_tiles = np.sum(thumbnail_grey)
            print('Slide shape', np.array(thumbnail).shape, 
                'No. of tumor tiles', num_tumor_tiles)
            # plt.imshow(thumbnail_grey)

        # merge tissue and ground truth tiles
        patches_y = pd.DataFrame(pd.DataFrame(thumbnail_grey).stack())
        patches_y['is_tumor'] = patches_y[0] > 0
        patches_y.drop(0, axis=1, inplace=True)

        slide_info = pd.concat([patches, patches_y], axis=1)


        ###
        ### open fixation mask file and convert to patch_size
        ###
        for weight_threshold in WEIGHT_THRESHOLDS:
            mask_file = osp.join(GAZE_MASK_PATH, weight_threshold, '%s.npy' % selection[0])
            if not os.path.exists(mask_file):
                print('Warning: mask file %s not exist!' % mask_file)
                continue
            mask = np.load(mask_file)

            # new_h, new_w = mask.shape[0]/(PATCH_SIZE/32), mask.shape[1]/(PATCH_SIZE/32)
            # mask_new = np.array(mask_img.resize((int(new_w), int(new_h))))
            # if mask_new.shape[0]!=thumbnail_grey.shape[0] or \
            #         mask_new.shape[1]!=thumbnail_grey.shape[1]:
            #     print('eyetrack tiles ', mask_new.shape[0], mask_new.shape[1])
            #     continue

            # resize based on PATCH_SIZE - force to be same as openslide size
            mask_img = Image.fromarray(mask)
            mask_new = np.array(mask_img.resize((thumbnail_grey.shape[1], thumbnail_grey.shape[0])))
            
            ##
            ## merge to tissue and GT dataframe
            ##
            fixt = pd.DataFrame(pd.DataFrame(mask_new).stack())
            fixt['in_fixation'] = fixt[0] > 0        
            fixt.drop(0, axis=1, inplace=True)

            # skip if mask region is too small and result in 0 cell
            if fixt['in_fixation'].sum() == 0:
                print('Warning: threshold %s has no tiles' % weight_threshold)
                continue

            tiles_info = pd.concat([slide_info, fixt], axis=1) # by column

            """
            ##
            ## calculate fixation overlap another way
            ##
            num_fixation_tiles = np.sum(mask_new)
            print('No. of fixation tiles at threshold %s is %d' % (weight_threshold, 
                                                                num_fixation_tiles))
            p_fixation = num_fixation_tiles/(mask_new.shape[0] * mask_new.shape[1])
            print('fixation area proportion is', p_fixation)

            # calculate overlap
            union = np.sum(mask_new | thumbnail_grey)
            inter = np.sum(mask_new & thumbnail_grey)
            overlap = inter / union
            print('IOU is', overlap)
            """

            ##
            ## calculate overlap stat
            ##
            union = tiles_info[(tiles_info['in_fixation']==True) |
                            (tiles_info['is_tumor']==True)]
                # do not use 'is_tissue' here; will judge in extraction
            
            ## save this tile details for later extraction
            fixt_tiles = union[union['in_fixation']==True]
            fixt_tiles['tile_loc'] = list(fixt_tiles.index)
            fixt_tiles.reset_index(inplace=True, drop=True)
            fixt_tiles['threshold'] = weight_threshold
            sample_fixation = pd.concat([sample_fixation, fixt_tiles], ignore_index=True)

            num_fixation_tiles = len(fixt_tiles)

            inter = len(union[(union['in_fixation']==True) &
                        (union['is_tumor']==True)])
            overlap = inter / len(union)
            print('union %d intersection %d iou %f' % (len(union), inter, overlap))

            # save summary results
            pid = selection[0].split('C')[0]
            iou_fixation = inter/num_fixation_tiles
            iou_gt = inter/num_tumor_tiles

            overlaps.append([selection[0], pid, selection[1], num_tumor_tiles, 
                            weight_threshold, num_fixation_tiles,  
                            inter, overlap, iou_fixation, iou_gt])

    # output results
    df_tissue_thresh = pd.DataFrame(data=tissue_thresh, 
                       columns=['slide_name', 'thresh_min', 'thresh_max'])
    df_tissue_thresh.to_csv('./fixation_tiles/tissue_threshold.csv', index=False)

    sample_fixation.to_csv('./fixation_tiles/fixation_tile_details.csv', index=False)

    df_res = pd.DataFrame(data=overlaps, 
                        columns=['maskID', 'pID', 'imageID', 'n_tumor_tiles',
                                'threshold', 'n_fixt_tiles', 
                                'n_fixt_tumor', 'iou', 'p_iou_fixt', 'p_iou_gt'])
    df_res.to_csv('./fixation_tiles/gt_fixation_stat_%d.csv' % PATCH_SIZE, index=False)

    return True



### plot fixation tile information
def plotFixationStat(df_res):
    
    df_res = pd.read_csv('./fixation_tiles/gt_fixation_stat_224.csv')
    df_sel = df_res.groupby('threshold').agg({'n_fixt_tiles':'sum', 
                                            'n_fixt_tumor':'sum'}).reset_index()
    df_sel = pd.melt(df_sel, id_vars =['threshold'], value_vars =['n_fixt_tiles','n_fixt_tumor'])

    fig, ax = plt.subplots(figsize=(5,4))
    sns.barplot(data=df_sel, x='threshold', y='value', hue='variable')
    plt.legend(title='Type', )
    plt.title('Number of fixation tiles')
    plt.savefig('results_plots/num_fixation_tiles.png', dpi=200, bbox_inches='tight')



### crop a tumor or non-tumor tile
def crop_a_tile(slide, x, y, thresh):
    # note y before x in extration
    rgb_image = slide.read_region((y*PATCH_SIZE, x*PATCH_SIZE), 0, CROP_SIZE)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])

    return (rgb_array, rgb_binary)



### extracting tiles
def extract_tiles():
    
    # record tiles tissue info
    tile_tissue_info = []

    for slide_name in tissue_thresh['slide_name']:

        df_tiles = sample_fixation[sample_fixation['slide_path'].str.contains(slide_name)]
        if len(df_tiles) == 0:
            continue
    
        slide_file = df_tiles['slide_path'].values[0]
        thresh = tissue_thresh[tissue_thresh['slide_name']==slide_name]
        thresh_min = thresh['thresh_min'].values[0]
        thresh_min = np.array(list(map(int,thresh_min.split(','))))
        thresh_max = thresh['thresh_max'].values[0]
        thresh_max = np.array(list(map(int,thresh_max.split(','))))

        # open slide image
        with openslide.open_slide(slide_file) as slide:
            
            # get 'tumor' patches
            m = 0
            for tile_loc in df_tiles[df_tiles['is_tumor']==True]['tile_loc']:
                x = int(tile_loc.split(', ')[0][1:])
                y = int(tile_loc.split(', ')[1][:-1])
                res = crop_a_tile(slide, x, y, (thresh_min, thresh_max))
                p_tissue = cv2.countNonZero(res[1])/(CROP_SIZE[0]*CROP_SIZE[1])
                if (p_tissue > MIN_TISSUE_THRESH):    
                    if (m) % 100 == 0:
                        print('tumor', m)
                    imsave(osp.join(output_path_tumor, '%s_%d_%d.png' % (slide_name, x, y)), res[0])
                    m = m + 1
                # document tile tissue info
                tile_tissue_info.append([slide_name, x, y, 'tumor', p_tissue])
                
            # get normal patches
            n=0
            for tile_loc in df_tiles[df_tiles['is_tumor']==False]['tile_loc']:
                x = int(tile_loc.split(', ')[0][1:])
                y = int(tile_loc.split(', ')[1][:-1])
                res = crop_a_tile(slide, x, y, (thresh_min, thresh_max))
                p_tissue = cv2.countNonZero(res[1])/(CROP_SIZE[0]*CROP_SIZE[1])
                if (p_tissue > MIN_TISSUE_THRESH): 
                    if (n) % 100 == 0: 
                        print('normal', n)
                    imsave(osp.join(output_path_normal, '%s_%d_%d.png' % (slide_name, x, y)), res[0])
                    n = n + 1
                # document tile tissue info
                tile_tissue_info.append([slide_name, x, y, 'normal', p_tissue])
                
    df_tile = pd.DataFrame(data=tile_tissue_info, 
                           columns=['slide_name', 'x', 'y', 'type', 'p_tissue'])
    df_tile.to_csv('./fixation_tiles/fixation_tile_tissue_info.csv', index=False)



# %%
###############
# main code
###############

### global variables
PATCH_SIZE = 224
CROP_SIZE = (PATCH_SIZE, PATCH_SIZE)
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training/all/'
SLIDE_TRUTH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training_masks/'
GAZE_MASK_PATH = '/fs/ess/PAS1575/Dataset/new_data/fixation_masks/fixation_reduction/'
TRAIN_SLIDES_PARTICIPANT = './experimentData/trainSlidesCorrectDaignosis.csv'

WEIGHT_THRESHOLDS = ['0.1', '0.3','0.5','0.7','0.9']

file_tile_details = './fixation_tiles/fixation_tile_details.csv'
file_tissue_thresh = './fixation_tiles/tissue_threshold.csv'
fixation_threshold = 0.1

MIN_TISSUE_THRESH = 0.2     # the min portion of tissue in a tile required for extraction


# obtain slide tissue and fixation data if not yet
if (not osp.exists(file_tile_details)) or (not osp.exists(file_tissue_thresh)):
    done = calculateFixationTiles()
    if not done:
        print('Failed to get fixation tile info!')
        sys.exit()
    else:
        print('Fixation tile info successfully obtained!')

# read slide tissue and fixation data
tissue_thresh = pd.read_csv(file_tissue_thresh)
sample_fixation = pd.read_csv(file_tile_details)
sample_fixation = sample_fixation[sample_fixation['threshold']==fixation_threshold]

# create output path
output_path_tumor = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMask/fixation/tumor/'
os.makedirs(output_path_tumor, exist_ok=True)
output_path_normal = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMask/fixation/normal/'
os.makedirs(output_path_normal, exist_ok=True)

# extract...
extract_tiles()






# %%
