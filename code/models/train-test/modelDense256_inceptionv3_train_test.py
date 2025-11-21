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


ROOT_PATH = '/fs/scratch/PAS1575/Pathology/CAMELYON16/individualMaskStainNorm'
ERROR_NAME = 'ERROR'



################### define model and objective #################
def build_model():
    # Pick a CNN model as feature extractor
    feature_generator = keras.applications.InceptionV3(include_top=False, input_shape=(224, 224, 3))

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
    sgd = keras.optimizers.legacy.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # compile the model before train
    model.compile(loss='binary_crossentropy', 
                optimizer=sgd,    # default adam is worse
                steps_per_execution=10,     # evaluate and output every 10 batches/steps
                metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])  
                # 'accuracy' appears to be equal to 'binary_accuracy'

    model.summary()
    
    return model



def model_train(model, train_name):
    ### construct datasets and dataloader
    seed = train_name.split('_')[-1].replace('s','')
    np.random.seed(int(seed))

    # get tumor train tiles
    data_path_train_tumor = osp.join(ROOT_PATH, 'train/tumor/%s' % train_name)
    data_tumor_train = glob.glob(osp.join(data_path_train_tumor, '*.png'))
    data_tumor_train = [[x, 1] for x in data_tumor_train]
    num_tumor_train = len(data_tumor_train)
    if num_tumor_train < 76000:
        print('train tumor tiles %d not complete!' % num_tumor_train)
        return ERROR_NAME
    else:
        print('train tumor tiles %d' % num_tumor_train)
   
    # get the training slides normal dataset
    data_path_train_normal_trainset = osp.join(ROOT_PATH, 'train/normal/trainset_s%s' % seed)
    data_normal_train_trainset = glob.glob(osp.join(data_path_train_normal_trainset, '*.png')) 
    data_normal_train_trainset = [[x, 0] for x in data_normal_train_trainset]
    num_normal_train_trainset = len(data_normal_train_trainset)
    if num_normal_train_trainset < 79000:
        print('train normal trainset s%s tiles %d not complete!' % (seed, num_normal_train_trainset))
        return ERROR_NAME
    else:
        print('train normal trainset s%s tiles %d' % (seed, num_normal_train_trainset))

    # get validation dataset
    data_path_val_tumor = osp.join(ROOT_PATH, 'validation/tumor/%s' % train_name)
    data_tumor_val = glob.glob(osp.join(data_path_val_tumor, '*.png'))
    data_tumor_val = [[x, 1] for x in data_tumor_val]
    num_tumor_val = len(data_tumor_val)
    if num_tumor_val < 18000:
        print('validation tumor tiles %d not complete!' % num_tumor_val)
        return ERROR_NAME
    else:
        print('validation tumor tiles %d' % num_tumor_val)

    # get the training slides normal dataset
    data_path_val_normal_trainset = osp.join(ROOT_PATH, 'validation/normal/trainset_s%s' % seed)
    data_normal_val_trainset = glob.glob(osp.join(data_path_val_normal_trainset, '*.png')) 
    data_normal_val_trainset = [[x, 0] for x in data_normal_val_trainset]
    num_normal_val_trainset = len(data_normal_val_trainset)
    if num_normal_val_trainset < 18000:
        print('validation normal trainset s%s tiles %d not complete!' % (seed, num_normal_val_trainset))
        return ERROR_NAME
    else:
        print('validation normal trainset s%s tiles %d' % (seed, num_normal_val_trainset))

    # construct train and val
    train_data = data_tumor_train + data_normal_train_trainset
    val_data = data_tumor_val + data_normal_val_trainset

    # Parameters
    params = {'dim': (224,224,3),
            'batch_size': 32,
            'shuffle': True}

    # create Generators
    training_generator = DataGenerator(train_data, **params)
    validation_generator = DataGenerator(val_data, **params)

    # take a look at the fetched data
    x, y = training_generator.__getitem__(10)
    print(x.shape)

    # plot tile to check
    # import matplotlib.pyplot as plt
    # plt.imshow(x[31]+0.5)

    MODELFILE = 'inceptionv3_models/TNsep_%s.h5' % train_name

    # set steps per epoch and num epoches
    N_epochs = 25    # train size 160K with bs32 translates into 5000 steps for a full epoch
    steps_per_epoch = training_generator.N//params['batch_size']

    callbacks = [keras.callbacks.ModelCheckpoint(MODELFILE, monitor='val_binary_accuracy', verbose=1, 
                                                 save_best_only=True, mode='max',
                                                 save_weights_only=True)]   
                                        
    t0 = time.time()

    history = model.fit(training_generator,
                        steps_per_epoch = steps_per_epoch,
                        epochs=N_epochs,                        
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        verbose=True)

    # save history
    with open(MODELFILE.replace('.h5', '_history.pkl'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    print('Fitting done', (time.time()-t0)/3600, 'hours')

    return MODELFILE



################### prediction ################# 
def model_test(model, MODELFILE):
    # read in data
    tiles_tumor_test = glob.glob(osp.join(ROOT_PATH, 'test/tumor/*.png'))
    data_tumor_test = [[x, 1] for x in tiles_tumor_test]
    num_tumor_test = len(data_tumor_test)
    print('test tumor tiles %d' % num_tumor_test)

    tiles_normal_test = glob.glob(osp.join(ROOT_PATH, 'test/normal/*.png'))
    data_normal_test = [[x, 0] for x in tiles_normal_test]
    num_normal_test = len(data_normal_test)
    print('test normal tiles %d' % num_normal_test)
    
    test_data = np.array(data_tumor_test + data_normal_test, dtype='object')
    print(test_data.shape)

    # split into batches
    batch_size = 1000
    num_batch = len(test_data)//batch_size
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
    df_res.to_csv(MODELFILE.replace('models', 'testing').replace('.h5', '_test.csv'), index=False)

    target_names = ['normal', 'tumor']
    report = classification_report(ygt_all, ypred_all, target_names=target_names)
    print(report)

    # save output as csv
    report = classification_report(ygt_all, ypred_all, target_names=target_names,
                                output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(MODELFILE.replace('models', 'testing').replace('.h5', '_report.csv'), index=False)



# load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get train dataset folder name')
    parser.add_argument('-d', metavar='--subfolder', type=str, action='store', 
                        dest='dataset', required=True, 
                        help='Train dataset folder name, e.g., 16macro_s6')
    
    # Gather the provided arguments as an array.
    args = parser.parse_args()
    train_name = vars(args)['dataset']
    print('dataset folder name:', train_name)
    data_path = osp.join(ROOT_PATH, 'train/tumor/%s' % train_name)
    
    
    if not osp.exists(data_path):
        print('train dataset folder name <%s> does not exist!' % data_path)
    else:
        model = build_model()
        model_name = model_train(model, train_name)
        if model_name != ERROR_NAME:
            model_test(model, model_name)

