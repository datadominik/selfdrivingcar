import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import matplotlib
import random
import glob

from keras.models import Sequential
from keras.layers import Convolution2D, Lambda, Dense, Activation, Dropout, MaxPooling2D, Flatten, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
import seaborn as sns

### First I define several augmentation methods
### Those are used to enlarge the dataset and prevent overfitting

def aug_flip(img,y):
    ### flips image in horizontal direction
    ### flips steering angle as well
    img = np.copy(img)
    y = np.copy(y)
    img = np.fliplr(img)
    y *= (-1)
    return(img,y)

def aug_zoom(img,y):
    ### zooms out and pads empty space
    ### not used in final model
    img = np.copy(img)
    y = np.copy(y)
    shape = img.shape
    zoom_factor = np.random.uniform(low=0.89,high=0.98)
    img = cv2.resize(img,(0,0),fx=zoom_factor, fy=zoom_factor, interpolation = cv2.INTER_CUBIC)

    rows, cols, channels = img.shape
    tmp = []
    for channel in range(channels):
        tmp.append(np.pad(img[:,:,channel],(shape[0]-rows,shape[1]-cols),mode='reflect'))

    img = np.array(tmp)
    img = img.transpose(1,2,0)

    img = cv2.resize(img,(shape[1],shape[0]))
    return(img,y)

def aug_dirt(img,y):
    ### removes small fractions of the image and adds random noise
    ### similar to water drops on camera
    img = np.copy(img)
    y = np.copy(y)
    shape = img.shape
    noise_size = 25
    noise_count = np.random.randint(10)
    for count in range(noise_count):
        row = np.random.randint(shape[0]-noise_size)
        col = np.random.randint(shape[1]-noise_size)
        img[row:row+noise_size,col:col+noise_size] = np.random.randint(255,size=(noise_size,noise_size,3))
    return(img,y)

def aug_shad(img,y):
    ### adds partial shadows on the camera picture
    ### I've seen similar approaches in several Udacity posts on medium.com
    img = np.copy(img)
    y = np.copy(y)
    ### transform image to hsv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = img.astype("float16")
    triangle = np.ones((500,500))
    ### get triangle indices
    xx = np.triu_indices(triangle.shape[0])[0]
    yy = np.triu_indices(triangle.shape[1])[1]
    x_flip = random.choice([-1, 1])
    y_flip = random.choice([-1, 1])
    triangle[x_flip*xx,y_flip*yy] = 0.5
    ### rotate triangle to get different angles
    ### make size afterwards same as the image
    rotation = cv2.getRotationMatrix2D((250+np.random.randint(50),250),np.random.randint(90),1.0)
    triangle = cv2.warpAffine(triangle, rotation, triangle.shape,flags=cv2.INTER_LINEAR)
    triangle = triangle[img.shape[0]-int(img.shape[0]/2):img.shape[0]+int(img.shape[0]/2),img.shape[1]-int(img.shape[1]/2):img.shape[1]+int(img.shape[1]/2)]
    ### mulitply last hsv channel times the triangle to create shadow effect
    img[:,:,2] *= triangle
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return(img,y)

def aug_colo(img,y):
    ### changes color brightness
    img = np.copy(img)
    y = np.copy(y)

    factor = np.random.rand(1)
    factor *= random.choice([-1, 1])
    img = cv2.add(img, np.array([factor]))
    return(img,y)




def train_generator(data,batch_size,train=True):
    ### a generator is used here to:
    ### 1. apply the augmentation during training
    ### 2. save memory by only loading batches of e.g. 32

    ### probabilities of augmentation methods
    p_save = 0.1
    p_flip = 0.5
    p_dirt = 0.3
    p_shad = 0.4
    p_colo = 0.5

    data = data.sample(frac=1).reset_index(drop=True)

    while 1:
        for i in range(0,len(data)-batch_size,batch_size):

            X_train = []
            y_train = []

            directions = ['center', 'left', 'right']

            for ii in range(i,i+batch_size):
                for direction in directions:
                    img = plt.imread("data/"+data.iloc[ii][direction].lstrip())

                    y = data.iloc[ii].steering

                    ### use left and right camera images since we otherwise have a too strong focus on 0 steering angles
                    if ((direction=='left')):
                        y += 0.25
                    if((direction=='right')):
                        y -= 0.25

                    ### only augmentate for training, not when generator is used on validation data
                    if(train):
                        if np.random.rand(1) < p_flip:
                            img, y = aug_flip(img, y)
                        if np.random.rand(1) < p_dirt:
                            img, y = aug_dirt(img, y)
                        if np.random.rand(1) < p_shad:
                            img, y = aug_shad(img, y)

                img = cv2.resize(img,(160, 80))
                img = img/255.

                X_train.append(img)
                y_train.append(y)

            X_train = np.asarray(X_train)

            X_train = X_train.astype("float16")
            y_train = np.asarray(y_train)

            yield (X_train, y_train)

def Nvidia():
    ### Simple convolutional neural network based on the approach of Nvidia
    ### Only little details are changed, for example:
    ### - Dense layers have different node counts
    ### - Dropout added to prevent overfitting
    model = Sequential()

    model.add(Convolution2D(24, 3, 3,input_shape=(80,160,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Convolution2D(36, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Convolution2D(48, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    return model


driving_data = pd.read_csv("data/driving_log.csv")
driving_data = driving_data.sample(frac=1).reset_index(drop=True)
driving_data.head()

sns.distplot(driving_data.steering)

model=Nvidia()
model.compile(loss="mse", optimizer="adam")
checkpoint = ModelCheckpoint('best_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

### split data into train and validation samples
train_data = driving_data.iloc[0:int(len(driving_data)*0.9)]
val_data = driving_data.iloc[int(len(driving_data)*0.9):len(driving_data)]

model.fit_generator(train_generator(data=train_data,batch_size=32),steps_per_epoch=len(train_data)/32,
nb_epoch=30,verbose=2,callbacks=[checkpoint],
                   validation_data=train_generator(data=val_data,batch_size=32,train=False),validation_steps=len(val_data)/32)
