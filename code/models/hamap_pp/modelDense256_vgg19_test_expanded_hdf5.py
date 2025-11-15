#############################################
#
# read normal tiles only from normal slides
#
#############################################

### Load libraries
import sys
sys.path.append('../')

import numpy as np
import glob
import os.path as osp

### Load keras libraries
from tensorflow import keras
import argparse as ap
import pandas as pd
import h5py

### set GPU
import os
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
    feature_generator = keras.applications.VGG19(include_top=False, input_shape=(224, 224, 3))

    ### define the mlp projection head
    MLP = keras.models.Sequential()
    MLP.add(keras.layers.Flatten(input_shape=feature_generator.output_shape[1:]))
    MLP.add(keras.layers.Dense(256, activation='relu', input_dim=(224, 224, 3)))
    MLP.add(keras.layers.Dropout(0.5))
    MLP.add(keras.layers.Dense(1, activation='sigmoid'))

    # define the model
    model = keras.Model(inputs=feature_generator.input,
                        outputs=MLP(feature_generator.output))

    # set optimizer
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # compile the model before train
    model.compile(loss='binary_crossentropy', 
                optimizer=sgd,    # default adam is worse
                steps_per_execution=10,     # evaluate and output every 10 batches/steps
                metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])  
                # 'accuracy' appears to be equal to 'binary_accuracy'

    model.summary()

    return model


def get_all_coordinates(hdf5_file):
    coordinates = []
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset):
            coordinates.append(name)
    hdf5_file.visititems(visit_func)
    return coordinates


################### prediction ################# 
def model_test_hdf5(MODELFILE, slide_name, hdf5_file_path):
    """
    Test the model using tiles stored in an HDF5 file.

    Args:
        model: The trained Keras model.
        MODELFILE: Path to the model file.
        hdf5_file_path: Path to the HDF5 file containing the tiles.
    """
    yhat_all = []  # to save predicted probability
    coords_all = [] # to save coordinates

    # load the model
    model = build_model()
    model.load_weights(MODELFILE)

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        coordinates = get_all_coordinates(hdf5_file)
        num_tiles = len(coordinates)
        print(f'Total number of tiles in HDF5 file: {num_tiles}')

        batch_size = 1024
        num_batches = (num_tiles + batch_size - 1) // batch_size  

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, num_tiles)
            batch_coords = coordinates[start_index:end_index]
            print('batch from %d to %d' % (start_index, end_index))

            # Prepare X
            X = np.empty((len(batch_coords), *(224, 224, 3)), dtype=np.single)
            for i, coord in enumerate(batch_coords):
                _, x_y = coord.split('/')
                x, y = x_y.split('_')
                coords_all.append((int(x), int(y)))
                img_array = np.array(hdf5_file[coord])
                img = img_array[:,:,:3]
                X[i,] = np.divide(img, 255.0, dtype=np.single) - 0.5

            # Predict yhat - probability
            yhat = model.predict(X)
            yhat_all.extend([x[0] for x in yhat])

    res = list(zip(yhat_all, coords_all))
    df_res = pd.DataFrame(res, columns=['yhat', 'coords'])
    df_res['x'] = df_res['coords'].apply(lambda x: x[0])
    df_res['y'] = df_res['coords'].apply(lambda x: x[1])
    df_res.drop(columns=['coords'], inplace=True)
    df_res.to_csv(f'./whole_slide_prediction/{slide_name}.csv', index=False)
    print('Prediction results saved to CSV file.')


# load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get train model name')
    parser.add_argument('-s', metavar='--slide', type=str, action='store', 
                        dest='slide_name', required=True, 
                        help='Slide name, e.g., normal_006')
    
    # Gather the provided arguments as an array.
    args = parser.parse_args()
    slide_name = vars(args)['slide_name']
    print('slide_name:', slide_name)
    model_name = './vgg19_models/TNsep_tumor_s26.h5'
    
    if not osp.exists(model_name):
        print('Model <%s> does not exist!' % model_name)
    else:
        hdf5_file_path = f'/fs/ess/PAS1575/stain_norm_tiles_h5py/{slide_name}.hdf5'
        if osp.exists(hdf5_file_path):
            model_test_hdf5(model_name, slide_name, hdf5_file_path)
        else:
            print('HDF5 file <%s> does not exist!' % hdf5_file_path)


