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
from sklearn.model_selection import train_test_split
import argparse as ap
import ast
import time
import sys

# Flush the stdout buffer in real time; MUST use -u in qsub/sbatch file
sys.stdout.flush()  

    
### global variables
PATCH_SIZE = 224
CROP_SIZE = (PATCH_SIZE, PATCH_SIZE)
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training/all/'
TRAIN_SLIDES_PARTICIPANT = './experimentData/trainSlidesCorrectDaignosis.csv'
OUTPUT_PATH = '/fs/scratch/PAS1575/Pathology/CAMELYON16/tissueTiles/'
TISSUE_CONTENT = 0.2    # the min proportion of tissue requires for a tile 


### randomly crop a tile from within the tissue lesion box
def crop_tissue(slide, tile_loc, minhsv, maxhsv):

    x = tile_loc[1] * PATCH_SIZE    # column
    y = tile_loc[0] * PATCH_SIZE    # row
    rgb_image = slide.read_region((x, y), 0, CROP_SIZE)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, minhsv, maxhsv)

    return (rgb_array, rgb_binary)
    


### run the extraction
def main(start, end):
    
    # read slide and tissue thresholds
    df_thresh = pd.read_csv('./experimentData/oursSlidesTissueThresholds.csv')
    slide_names = df_thresh['slide_name'].values[start:end]

    # read tissue tile data
    df_tiles = pd.read_csv('./experimentData/oursSlidesTissueTiles.csv')

    t_start = time.time()

    for slide_name in slide_names:
        print(slide_name)
        t0 = time.time()

        # create output path
        output_path = f'/fs/scratch/PAS1575/Pathology/CAMELYON16/tissueTiles/{slide_name}' 
        os.makedirs(output_path, exist_ok=True)

        # get thresholds
        df_th = df_thresh[df_thresh['slide_name']==slide_name]
        minhsv = df_th['minhsv'].values[0]
        maxhsv = df_th['maxhsv'].values[0]
        # need to be separated by comma before ast.literal_eval()
        minhsv = np.array(ast.literal_eval(minhsv.replace('  ', ',').replace(' ', ',')))    # sometimes two spaces
        maxhsv = np.array(ast.literal_eval(maxhsv.replace('  ', ',').replace(' ', ',')))
        
        # get tile locations for each slide
        tile_locs = df_tiles[df_tiles['slide_name']==slide_name]['tile_loc'].values
        print(f'\tContains {len(tile_locs)} tiles')

        # check tiles already in folder
        tiles_extracted = glob.glob(osp.join(output_path, '*.png'))
        tiles_extracted = [osp.basename(x) for x in tiles_extracted]
        print(f'\tAlready extracted {len(tiles_extracted)} tiles')
        print(f'\tNumber of tiles to extract {len(tile_locs)-len(tiles_extracted)}')

        # open the slide
        with openslide.open_slide(osp.join(SLIDE_PATH, f'{slide_name}.tif')) as slide:

            # extract all tiles
            n = 0
            for tile_loc in tile_locs:
                tile_loc = ast.literal_eval(tile_loc)
                tile_name = f'{slide_name}_{tile_loc[1]}_{tile_loc[0]}.png'
                if not tile_name in tiles_extracted:
                    img, img_binary = crop_tissue(slide, tile_loc, minhsv, maxhsv)
                    if (cv2.countNonZero(img_binary) > CROP_SIZE[0]*CROP_SIZE[1]*TISSUE_CONTENT):    
                        if (n+1) % 1000 == 0: 
                            print(f'\t{n+1}')
                        imsave(osp.join(output_path, tile_name), img)
                        n = n + 1
        print(f'\tExtraction done in {(time.time()-t0)/3600:.2f} hours')

    print(f'\nAll extraction done in {(time.time()-t_start)/3600:.2f} hours')



# %% 
### load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get dataset seed and eyetrack threshold')
    parser.add_argument('-start', type=int, action='store', 
                        dest='start', required=True, 
                        help='start slide, i.e., from 0')
    parser.add_argument('-end', type=int, action='store', 
                        dest='end', required=True, 
                        help='end slide, i.e., from 1')

    # read slide and count number
    df_thresh = pd.read_csv('./experimentData/oursSlidesTissueThresholds.csv')
    num_slides = len(df_thresh)
    
    args = parser.parse_args()
    start = vars(args)['start']
    end = vars(args)['end']

    if start>(num_slides-1):
        print(f'start {start} cannot exceed number of slides {num_slides}')
    elif end>num_slides:
        end = num_slides

    # extract from the start slide to end slide
    main(start, end)



# %%
