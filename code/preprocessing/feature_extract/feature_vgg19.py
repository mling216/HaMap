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
def model_test(model, num_each_type, stain=False):
    # read in data
    if stain==False:
        tiles_tumor = glob.glob(osp.join(ROOT_PATH, 'tumor/*.png'))
    else:
        tiles_tumor = glob.glob(osp.join(ROOT_PATH_STAIN, 'tumor/*.png'))
    
    if RANDOM_DATA_SEED > 0:
        random.seed(RANDOM_DATA_SEED)
    data_tumor = random.choices(tiles_tumor, k=num_each_type)

    tiles_normal = glob.glob(osp.join(ROOT_PATH, 'normal/*.png'))
    if RANDOM_DATA_SEED > 0:
        random.seed(RANDOM_DATA_SEED)
    data_normal = random.choices(tiles_normal, k=num_each_type)

    data_labels = ['Tumor'] * num_each_type + ['Normal'] * num_each_type

    data_extract = data_tumor + data_normal
    print('tumor tiles %d normal tiles %d' % (len(data_tumor), len(data_normal)))

    # prepare X
    X = np.empty((len(data_extract), *(224,224,3)), dtype=np.single)
    for i, tile in enumerate(data_extract):
        img = imread(tile)
        img = img[:,:,:3]   # testing images are 4 channels
        X[i,] = np.divide(img, 255.0, dtype=np.single) - 0.5

    # predict yhat - probability
    features = model.predict(X)
    print('Features extracted!')

    return features, data_labels



################### plot tsne ################# 

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def tsne_plot(features, data_labels, num_each_type, stain):
    # calculate tSNE - set seed for reproucible results
    tsne = TSNE(n_components=2, random_state=66).fit_transform(features)
    print('tSNE shape', tsne.shape)
    
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # save results to dataframe
    df = pd.DataFrame([])
    df['x'], df['y'], df['label'], df['feature'] = tx, ty, data_labels, list(features)
    df.to_csv('./fixation_tiles/feature_tsne_%d_tiles.csv' % num_each_type)

    # make a matplotlib plot
    fig, ax = plt.subplots(figsize=(8,8))
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
        fname = ('./fixation_tiles/plot_tsne_%d_tiles_s%d_stain.png' % 
                 (num_each_type*2, RANDOM_DATA_SEED))
    else:
        fname = ('./fixation_tiles/plot_tsne_%d_tiles_s%d.png' % 
                 (num_each_type*2, RANDOM_DATA_SEED))
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print('image saved!')



################## main code ###############

ROOT_PATH = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMask/fixation'
ROOT_PATH_STAIN = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMaskStainNorm/fixation'
RANDOM_DATA_SEED = 8 # 0 indicates random

def main(num_each_type=100):
    model = build_model()

    stain=False
    features, data_labels = model_test(model, num_each_type, stain=stain)
    tsne_plot(features, data_labels, num_each_type, stain=stain)

    stain=True
    features, data_labels = model_test(model, num_each_type, stain=stain)
    tsne_plot(features, data_labels, num_each_type, stain)



# load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get number of tiles for each type')
    parser.add_argument('-n', metavar='--number', type=int, action='store', 
                        dest='num_each_type', default=100,
                        help='number of tiles for each type, e.g., 100')
    
    parser.add_argument('-r', metavar='--random', type=int, action='store', 
                        dest='random_seed', default=100,
                        help='random seed for tile selection, e.g., 6 or 1234')
        
    # Gather the provided arguments as an array.
    args = parser.parse_args()
    num_each_type = vars(args)['num_each_type']
    RANDOM_DATA_SEED = vars(args)['random_seed']
    print('number of tiles for each type:', num_each_type)
    print('random seed for tile selection:', RANDOM_DATA_SEED)
    main(num_each_type)


