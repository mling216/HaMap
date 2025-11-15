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


### global variables
PATCH_SIZE = 224
SLIDE_PATH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training/all/'
SLIDE_TRUTH = '/fs/ess/PAS1575/Dataset/CAMELYON16/training_masks/'
GAZE_MASK_PATH = '/fs/ess/PAS1575/Dataset/new_data/fixation_masks/fixation_reduction/'
TRAIN_SLIDES_PARTICIPANT = './experimentData/trainSlidesCorrectDaignosis.csv'
weight_thresholds = ['0.1', '0.3','0.5','0.7','0.9']



# %%
### get train tumor slides
df_slides = pd.read_csv(TRAIN_SLIDES_PARTICIPANT)
df_slides = df_slides[df_slides['diagnosis']!='Benign']
mask_names = df_slides['maskName'].to_list()

# save selections in a list
slides_to_test = []
for mask in mask_names:
    slides_to_test.append([mask, df_slides.loc[df_slides['maskName']==mask, 'imageName'].values[0]])
# print(slides_to_test)



# %%
### calculate overlaps
overlaps = []
for selection in slides_to_test:
    ############# diagnosis on a specific slide ############
    # if selection[1]!='tumor_001':
    #     continue
    ############# diagnosis on a specific slide ############

    print(selection[0])

    slide_file = SLIDE_TRUTH + selection[1] +'_mask.tif'
    with openslide.open_slide(slide_file) as slide:
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / PATCH_SIZE, 
                    slide.dimensions[1] / PATCH_SIZE))

        thumbnail_grey = np.array(thumbnail.convert('L'))/255.0
        thumbnail_grey = thumbnail_grey.astype(int)
        print('slide shape', np.array(thumbnail).shape, 
              'No. of tumor tiles', np.sum(thumbnail_grey))
        # plt.imshow(thumbnail_grey)

    
    # open mask file and convert to patch_size
    for weight_threshold in weight_thresholds:
        mask_file = osp.join(GAZE_MASK_PATH, weight_threshold, '%s.npy' % selection[0])
        if not os.path.exists(mask_file):
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
        
        ############# diagnosis on the gaze mask ############
        # print('%s (%s)' % (selection[0], selection[1]), 'number of 32 tiles is', mask.shape[0] * mask.shape[1])
        print('No. of tumor tiles at threshold %s is %d' % (weight_threshold, np.sum(mask_new)))
        p = np.sum(mask_new)/(mask_new.shape[0] * mask_new.shape[1])
        print('tumor area proportion is', p)
        # plt.imshow(mask)
        ############# diagnosis on the gaze mask ############

        # calculate overlap
        union = np.sum(mask_new | thumbnail_grey)
        inter = np.sum(mask_new & thumbnail_grey)
        overlap = inter / union
        print('IOU is', overlap)

        # save results
        pid = selection[0].split('C')[0]
        overlaps.append([pid, selection[1], weight_threshold, 
                         'IoU', overlap])
        if np.sum(mask_new) > 0:
            overlaps.append([pid, selection[1], weight_threshold, 
                            'IoU_%_fixation', inter/np.sum(mask_new)])
        overlaps.append([pid, selection[1], weight_threshold, 
                         'IoU_%_GT', inter/np.sum(thumbnail_grey)])
       
        
# output results
df_res = pd.DataFrame(data=overlaps, columns=['pID', 'imageID', 'threshold', 'metric', 'value'])
df_res.to_csv('fixation_tiles/gt_fixation_overlap_%d.csv' % PATCH_SIZE, index=False)
df_res



# %%
### plot results
import seaborn as sns

sns.barplot(data=df_res, x='threshold', y='value', hue='metric', 
            capsize=0.05, width=0.7)
plt.title('Eyetracking fixation and groundtruth overlap')
plt.legend(title='Overlap', bbox_to_anchor=(1.0,1.02))
plt.savefig('results_plots/fixation_iou_groundtruth.png', dpi=200, bbox_inches='tight')



# %%
