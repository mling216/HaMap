# %%
import numpy as np
import tensorflow as tf
from imageio import imread


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, dataArray, batch_size=32, dim=(224,224,3), shuffle=True):
    
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.data = dataArray
        self.N = len(self.data)
        self.on_epoch_end()

    def __len__(self):
        # the number of batches per epoch
        return int(np.floor(self.N / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.dim), dtype=np.single)
        y = np.empty((self.batch_size), dtype=np.single)

        # Generate data
        for i, ID in enumerate(indexes):
            img = imread(self.data[ID][0])    # path to tile image (224, 224, 3)
            # print(img.shape)
            img = img[:,:,:3]   # testing images are 4 channels
            X[i,] = np.divide(img, 255.0, dtype=np.single) - 0.5
            y[i] = self.data[ID][1]  # label
        
        return X, y
        # keras.utils.to_categorical(y, num_classes=self.n_classes), 
        # usually for multi-class

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.N)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# %%
