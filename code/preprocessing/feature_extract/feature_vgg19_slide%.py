#############################################
#
# extract tile features using pre-trained model
#
#############################################

# %%
### Load libraries
import numpy as np
import glob
import os.path as osp
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

### Load keras libraries
from tensorflow import keras
import argparse as ap
from sklearn.metrics import classification_report
import pandas as pd
from imageio import imread

### set GPU
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    try: 
        tf.config.experimental.set_memory_growth(d, True) 
    except: 
        # Invalid device or cannot modify virtual devices once initialized. 
        pass 



################### define model and objective #################
def build_model():
    # Pick a CNN model as feature extractor
    base_model = keras.applications.VGG19(include_top=False, input_shape=(224, 224, 3))
    h = keras.layers.GlobalAveragePooling2D()(base_model.output)

    # define the model
    model = keras.Model(inputs=base_model.inputs, outputs=h)
    model.summary()
    
    return model



################### extract features ################# 
def model_test(model, percent, stain=False):
    # read in data
    if stain==False:
        tiles_tumor = glob.glob(osp.join(ROOT_PATH, 'tumor/*.png'))
        tiles_normal = glob.glob(osp.join(ROOT_PATH, 'normal/*.png'))
    else:
        tiles_tumor = glob.glob(osp.join(ROOT_PATH_STAIN, 'tumor/*.png'))
        tiles_normal = glob.glob(osp.join(ROOT_PATH_STAIN, 'normal/*.png'))
    
    # select the "percent" of test slides
    list_slides = pd.read_csv('./fixation_tiles/tissue_threshold.csv')
    if RANDOM_DATA_SEED > 0:
        random.seed(RANDOM_DATA_SEED)
    test_slides = random.choices(list_slides['slide_name'].to_list(), 
                   k=round(len(list_slides)*percent/100))

    # get tiles
    data_tumor = [x for x in tiles_tumor if any(s in osp.basename(x) for s in test_slides)]
    data_normal = [x for x in tiles_normal if any(s in osp.basename(x) for s in test_slides)]
    data_extract = data_tumor + data_normal
    print('tumor tiles %d normal tiles %d' % (len(data_tumor), len(data_normal)))

    ### set tile labels
    data_labels = ['Tumor'] * len(data_tumor) + ['Normal'] * len(data_normal)

    ### save tile information
    df = pd.DataFrame([])
    df['tile'], df['label'] = data_extract, data_labels
    if stain:
        df.to_csv('./fixation_tiles/mini_train_set_p%d_s%d_t%d_n%d.csv' % 
                  (percent, RANDOM_DATA_SEED, len(data_tumor), len(data_normal)))

    # prepare X
    X = np.empty((len(data_extract), *(224,224,3)), dtype=np.single)
    for i, tile in enumerate(data_extract):
        img = imread(tile)
        img = img[:,:,:3]   # testing images are 4 channels
        X[i,] = np.divide(img, 255.0, dtype=np.single) - 0.5

    # predict yhat - probability
    features = model.predict(X)
    print('Features extracted!')

    return features, data_extract, data_labels



################### plot tsne ################# 

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def tsne_plot(features, data_tiles, data_labels, percent, stain):
    # calculate tSNE - set seed for reproucible results
    tsne = TSNE(n_components=2, random_state=RANDOM_DATA_SEED).fit_transform(features)
    print('tSNE shape', tsne.shape)
    
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # save results to dataframe
    df = pd.DataFrame([])
    # df['x'], df['y'], df['label'], df['feature'] = tx, ty, data_labels, list(features)
    df['x'], df['y'], df['tile'], df['label'] = tx, ty, data_tiles, data_labels
    if stain:
        df.to_csv('./fixation_tiles/tsne_p%d_s%d_stain.csv' % (percent, RANDOM_DATA_SEED))
    else:
        df.to_csv('./fixation_tiles/tsne_p%d_s%d.csv' % (percent, RANDOM_DATA_SEED))

    # make a matplotlib plot
    fig, ax = plt.subplots(figsize=(12,12))
    markers = {'Tumor':'s','Normal':'x'} # '^','*','x','_','2','o','P','D','o'}
    
    for label in ['Tumor','Normal']:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(data_labels) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # add a scatter plot with the corresponding color and label        
        ax.scatter(current_tx, current_ty, label=label, alpha=1.0, marker=markers[label])
    
    # build a legend using the labels we set previously
    # ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
    ax.legend(loc='best')
    if stain:
        fname = ('./fixation_tiles/plot_tsne_p%d_s%d_stain.png' % 
                 (percent, RANDOM_DATA_SEED))
    else:
        fname = ('./fixation_tiles/plot_tsne_p%d_s%d.png' % 
                 (percent, RANDOM_DATA_SEED))
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print('image saved!')



################## main code ###############

ROOT_PATH = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMask/fixation'
ROOT_PATH_STAIN = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMaskStainNorm/fixation'
RANDOM_DATA_SEED = 8 # 0 indicates random

def main(percent):
    model = build_model()

    stain=False
    features, data_tiles, data_labels = model_test(model, percent, stain=stain)
    tsne_plot(features, data_tiles, data_labels, percent, stain=stain)
    
    stain=True
    features, data_tiles, data_labels = model_test(model, percent, stain=stain)
    tsne_plot(features, data_tiles, data_labels, percent, stain=stain)
    


# %%
### data check
# RANDOM_DATA_SEED = 35   # seed 35 having 2217,4468 tiles
# model_test(None, 10, stain=False)



# %% load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get number of tiles for each type')
    parser.add_argument('-p', metavar='--proportion', type=int, action='store', 
                        dest='percent', default=10,
                        help='% of slides to extract from, e.g., 10')
    
    parser.add_argument('-r', metavar='--random', type=int, action='store', 
                        dest='random_seed', default=35,
                        help='random seed for tile selection, e.g., 6 or 1234')
        
    # Gather the provided arguments as an array.
    args = parser.parse_args()
    percent = vars(args)['percent']
    RANDOM_DATA_SEED = vars(args)['random_seed']
    print('%d percent of slides to extract' % percent)
    print('random seed for tile selection:', RANDOM_DATA_SEED)
    main(percent)





