# %%
# from scipy.misc import imsave # need to be <=1.2.0
from imageio import imwrite as imsave
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
import cv2

from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

# SLIDE_PATH = "/fs/ess/PAS1575/Dataset/CAMELYON16/training/tumor"
SLIDE_PATH = "/fs/ess/PAS1575/Dataset/CAMELYON16/testing/images"
slide_paths = glob.glob(osp.join(SLIDE_PATH, "*.tif"))  # change this for needed slides
slide_paths = sorted(slide_paths)
slide_paths.sort()

OUTPUT_PATH = "/fs/scratch/PAS1575/Pathology/CAMELYON16/nonOverlapTest/"


# %%
### build a dataframe of all the tissue patches
#
def get_slide_tiles(part, n_part):
    if part == 0:
        print("part number starts from 1")
        return None, None
    if part > n_part:
        print("part number exceeds total parts")
        return None, None
    
    batchs = (len(slide_paths) + n_part - 1) // n_part
    batch_start = (part-1)*batchs
    batch_end = min(part*batchs, len(slide_paths))
    print(f"Processing part {part} of {n_part}, slides from {batch_start} to {batch_end}")
    cur_slide_paths = slide_paths[batch_start:batch_end]
    
    sampletotal = pd.DataFrame([])
    thresholds = []
    for i, slide_path in enumerate(cur_slide_paths):
        if 'test_049' in slide_path:
            print("skipping test_049, which has issues")
            continue
        with openslide.open_slide(slide_path) as slide:
            thumbnail = slide.get_thumbnail((slide.dimensions[0] / 224, slide.dimensions[1] / 224))

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
            thresholds.append([os.path.basename(slide_path).split('.')[0], thresh])

            rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
            binary = rgbbinary / 255
            binary = binary.astype(np.uint8)

            patches = pd.DataFrame(pd.DataFrame(binary).stack())
            patches['is_tissue'] = patches[0]   # not ~patches[0]
            patches.drop(0, axis=1, inplace=True)
            patches["slide_name"] = os.path.basename(slide_path)

        # remove patches with no tissue
        patches = patches[patches.is_tissue == True]
        patches["tile_loc"] = list(patches.index)
        patches.reset_index(inplace=True, drop=True)

        sampletotal = pd.concat((sampletotal, patches), ignore_index=True)

    sampletotal.drop(columns='is_tissue', inplace=True)
    sampletotal.to_csv('train_tumor_slides_tiles.csv', index=False)
    print(len(sampletotal))
    print(sampletotal['slide_name'].value_counts())

    return sampletotal, thresholds


# %%
### real image patches generation

def gen_imgs(slide_name, samples):

    slide_name_pre = slide_name.split('.')[0]
    save_path = os.path.join(OUTPUT_PATH, slide_name_pre)
    os.makedirs(save_path, exist_ok=True)

    with openslide.open_slide(os.path.join(SLIDE_PATH,slide_name)) as slide:
        tiles = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)

        for idx, row in samples.iterrows():
            if idx % 1000 == 0:
                print(idx)

            try:  # some slides may have error with open_slide
                img = tiles.get_tile(tiles.level_count-1, row['tile_loc'][::-1])
                im = np.array(img)
                # add check on tile size
                if im.shape[0] != 224 or im.shape[1] != 224:
                    continue
                int1, int2 = row['tile_loc'][::-1]
                imsave(os.path.join(save_path, f"{slide_name_pre}_{int1}_{int2}.png"), im)

            except:
                print("deepzoom open error", slide_name, row['tile_loc'][::-1])

            # if idx==5:
            #     break



# %% find tissue and tiles for each slide

def extract_non_overlap_tiles(part, n_part):
    sampletotal, thresholds = get_slide_tiles(part, n_part)

    df_thresh = pd.DataFrame(thresholds, columns=['slide_name','tissue_thresh'])
    df_thresh.to_csv(f'test_slides_tissue_thresholds_{part}_of_{n_part}.csv', index=False)

    slides_list = sorted(sampletotal['slide_name'].unique())

    # cut tiles by slide
    for i, slide_name in enumerate(slides_list):
        samples = sampletotal[sampletotal['slide_name']==slide_name]
        samples.reset_index(inplace=True, drop=True)
        print(slide_name, len(samples))
        gen_imgs(slide_name, samples)

    print('job done!')


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, required=True, help='part number to process')
    parser.add_argument('--n_part', type=int, required=True, help='total number of parts')
    args = parser.parse_args()
    part = args.part
    n_part = args.n_part
    extract_non_overlap_tiles(part, n_part)
    

