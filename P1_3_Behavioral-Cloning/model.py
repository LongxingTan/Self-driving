#!/usr/bin/env python
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from scipy.stats import bernoulli
from scipy.ndimage import rotate
import tensorflow as tf
from keras import backend as K
#K.set_image_dim_ordering('tf')

correlation=0.23
batch_size=32

def load_data():
    data=pd.read_csv('./data/driving_log.csv',sep=',')
    x_central,y_central=data.iloc[:,0],data.iloc[:,3]
    x_left,y_left=data.iloc[:,1],data.iloc[:,3]+correlation
    x_right,y_right=data.iloc[:,2],data.iloc[:,3]-correlation
    x=np.hstack((x_central,x_left,x_right))
    y=np.hstack((y_central,y_left,y_right))
    X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    return X_train,X_test,y_train,y_test

def preprocess_images(img):
    #img=cv2.resize(img,(160,160))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def random_flip(img,angle):
    return np.fliplr(img), -1 * angle


def argument_images(img,angle):
    #img,angle=random_translate(img,angle,range_x=60,range_y=10)
    #img, angle = random_rotate(img, angle)
    #img,angle=random_shear(img,angle)
    img, angle = random_flip(img, angle)
    #img = random_brightness(img)
    #img = random_gamma(img)
    return img,angle

def create_image_file(x,y,is_train):
    x, y = shuffle(x, y)
    images, angles = [], []
    for image, angle in zip(x, y):
        image = cv2.imread('./data/' + image.strip())
        image = preprocess_images(image)

        if is_train:
            images.append(image)
            angles.append(float(angle))
            image, angle = argument_images(image, angle)
            images.append(image)
            angles.append(float(angle))
        else:
            images.append(image)
            angles.append(angle)
    return images,angles


def create_batch(x,y,batch_size=batch_size,is_train=True):
    images, angles = create_image_file(x, y, is_train)
    n_x = len(images)
    #print(np.array(images).shape, np.array(angles).shape)
    #print(np.array(images[0:64]).shape,np.array(angles[0:64]).shape)
    while True:
        batch_images,batch_angles=[],[]
        for i in range(0, n_x, batch_size):
            batch_image, batch_angle = images[i:i + batch_size], angles[i:i + batch_size]
            #batch_images.append(batch_image)
            #batch_angles.append(batch_angle)
            batch_image=np.array(batch_image)
            #batch_image=batch_image.reshape(batch_image.shape[0],160,320,3)
            batch_angle=np.array(batch_angle)
            yield (batch_image,batch_angle)



def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 -0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (1, 1))))
    #model.add(Lambda(lambda images: tf.image.resize_images(images, (66, 200))))
    model.add(Convolution2D(24, 5, 5, activation='relu',subsample=(2,2),border_mode='valid'))
    model.add(Convolution2D(36, 5, 5, activation='relu',subsample=(2,2),border_mode='valid'))
    model.add(Convolution2D(48, 5, 5, activation='relu',subsample=(2,2),border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='valid'))
    model.add(Convolution2D(64, 3, 3,activation='relu',border_mode='valid'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    #model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    return model

#X_train,X_test,y_train,y_test=load_data()



X_train,X_test,y_train,y_test=load_data()

def small():
    X_example = cv2.imread('./data/' + X_train[1000].strip())
    X_img_example = preprocess_images(X_example)
    print(X_img_example.shape)
    plt.imshow(X_img_example)
    plt.show()

print(X_train.shape)
train_batch=create_batch(x=X_train,y=y_train,batch_size=batch_size,is_train=True)
valid_batch=create_batch(x=X_test,y=y_test,batch_size=batch_size,is_train=False)



model=create_model()
history_object=model.fit_generator(train_batch,nb_epoch=5,samples_per_epoch=2*len(X_train),validation_data=valid_batch,nb_val_samples=len(X_test),verbose = 1)
model.save('model.h5')
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



