#############################################
#
# read normal tiles only from normal slides
#
#############################################

### Load libraries
import numpy as np
from myKerasDataset import DataGenerator
import pickle
import time
import glob
import os.path as osp

### Load keras libraries
from tensorflow import keras
import argparse as ap
from sklearn.metrics import classification_report
import pandas as pd
from imageio import imread

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


ROOT_PATH = '/fs/scratch/PAS1575/Pathology/CAMELYON16/eyeTrack224StainNorm'



################### define model and objective #################
def build_model():
    # Pick a CNN model as feature extractor
    feature_generator = keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))

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



################### prediction ################# 
def model_test(model, MODELFILE):
    # read in data
    tiles_tumor_test = glob.glob(osp.join(ROOT_PATH, 'test/tumor/*.png'))
    data_tumor_test = [[x, 1] for x in tiles_tumor_test]
    num_tumor_test = len(data_tumor_test)
    print('test tumor tiles %d' % num_tumor_test)

    tiles_normal_test_tumor = glob.glob(osp.join(ROOT_PATH, 'test/normal/test_tumor/*.png'))
    data_normal_test_tumor = [[x, 0] for x in tiles_normal_test_tumor]
    num_normal_test_tumor = len(data_normal_test_tumor)
    print('test normal tiles from tumor slides %d' % num_normal_test_tumor)
    
    # get the testing slides normal dataset
    tiles_normal_test = glob.glob(osp.join(ROOT_PATH, 'test/normal/test_normal/*.png'))
    data_normal_test = [[x, 0] for x in tiles_normal_test]
    num_normal_test = len(data_normal_test)
    print('test normal tiles from normal slides %d' % num_normal_test)

    test_data = np.array(data_tumor_test + data_normal_test_tumor + data_normal_test, dtype='object')
    print(test_data.shape)

    # split into batches
    batch_size = 1000
    if len(test_data)%batch_size == 0:
        num_batch = len(test_data)//batch_size 
    else:
        num_batch = len(test_data)//batch_size + 1
    print('num of batchs', num_batch)

    # load the model
    model.load_weights(MODELFILE)

    yhat_all = []   # to save predicted probability
    ygt_all = []    # to save ground truth
    ypred_all = []    # to save predicted labels (0 or 1)

    ### predict results by batches
    for bn in range(num_batch):
        
        num_start = bn*batch_size
        if bn==(num_batch-1):
            num_end = len(test_data)
        else:
            num_end = (bn+1)*batch_size
        print('batch from %d to %d' % (num_start, num_end-1))

        # prepare X
        X = np.empty((num_end-num_start, *(224,224,3)), dtype=np.single)
        for i, fn in enumerate(test_data[num_start : num_end, 0]):
            img = imread(fn)
            img = img[:,:,:3]   # testing images are 4 channels
            X[i,] = np.divide(img, 255.0, dtype=np.single) - 0.5

        # predict yhat - probability
        yhat = model.predict(X)

        # predict labels
        ypred = [1 if y>0.5 else 0 for y in yhat]

        # ground truth
        ygt = test_data[num_start : num_end, 1]

        ygt_all.extend(ygt)
        ypred_all.extend(ypred)

        # save yhats for ROC plot
        yhat_all.extend([x[0] for x in yhat])

    # save results and gt to file
    res = list(zip(yhat_all, ygt_all))
    df_res = pd.DataFrame(res, columns=['yhat','ytruth'])
    df_res.to_csv(MODELFILE.replace('models', 'testing').replace('.h5', '_small_test.csv'), index=False)

    target_names = ['normal', 'tumor']
    report = classification_report(ygt_all, ypred_all, target_names=target_names)
    print(report)

    # save output as csv
    report = classification_report(ygt_all, ypred_all, target_names=target_names,
                                output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(MODELFILE.replace('models', 'testing').replace('.h5', '_small_report.csv'), index=False)



# load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get train dataset folder name')
    parser.add_argument('-d', metavar='--subfolder', type=str, action='store', 
                        dest='dataset', required=True, 
                        help='Train dataset folder name, e.g., 16macro_s6')
    
    # Gather the provided arguments as an array.
    args = parser.parse_args()
    train_name = vars(args)['dataset']
    print('dataset/model name:', train_name)
    model_name = './resnet50_models/TNsep_%s.h5' % train_name
    
    if not osp.exists(model_name):
        print('Model <%s> does not exist!' % model_name)
    else:
        model = build_model()
        model_test(model, model_name)

