#Key point: so omportant set the initialized stand deviation of ofilter for CNN
import csv

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels


# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './train.p'
validation_file ='./valid.p'
testing_file = './test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(list(y_train)+list(y_test)+list(y_valid)))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import collections
import pandas as pd
import numpy as np
# Visualizations will be shown in the notebook.

plt.figure(1, figsize=(20, 20))
image_name=pd.read_csv('signnames.csv',sep=',')
for i in range(n_classes):
    index=np.argwhere(y_train==i)[0,0]
    id = y_train[index]
    plt.subplot(10,5,i+1)
    plt.title(image_name.iloc[id,1], fontsize=7)
    plt.imshow(X_train[index].squeeze())
#plt.show()

value_count=collections.Counter(y_train)
df = pd.DataFrame(list(value_count.items()), columns=['ClassId', 'Count'])
df=df.merge(image_name,on='ClassId',how='left',right_index=False)
df.sort_values(by ='Count',ascending =False,inplace=True)
print(df)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


import numpy as np
import cv2
import tensorflow as tf
import random

def normalize(x):
    #gray=[]
    #for i in x:
        #img=cv2.resize(i, (32,32))
        ##img = Image.fromarray(i.astype('uint8'), 'RGB')
        #img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.5))
        #img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))
        #gray.append(img)
    return np.array((x-128)/128)#.reshape(-1,32,32,1)


def add_noise(img, noise_level=1.0):
    noisy_image = img + noise_level * img.std() * np.random.random(img.shape)
    return noisy_image

def augmentation(x,y):
    aug_x,aug_y=[],[]
    for i in range(len(x)):
        aug_x.append(x[i])
        aug_y.append(y[i])
        #aug_x.append(tf.contrib.keras.preprocessing.image.random_rotation(x[i], 20, row_axis=0, col_axis=1, channel_axis=2))
        #aug_y.append(y[i])
        aug_x.append(tf.contrib.keras.preprocessing.image.random_shear(x[i], 0.2, row_axis=0, col_axis=1, channel_axis=2))
        aug_y.append(y[i])
        #aug_x.append(tf.contrib.keras.preprocessing.image.random_shift(x[i], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
        #aug_y.append(y[i])
        aug_x.append(tf.contrib.keras.preprocessing.image.random_zoom(x[i], (0.9, 0.9), row_axis=0, col_axis=1, channel_axis=2))
        aug_y.append(y[i])
        aug_x.append(add_noise(x[i], np.random.uniform(low=1.0, high=1.2, size=(1,))[0]))
        aug_y.append(y[i])
        aug_image,aug_label=batch_randomize(np.array(aug_x),np.array(aug_y),batch_size=128)
    return aug_image,aug_label


def batch_randomize(x, y, batch_size=128):
    # Generate the permutation index array.
    permutation = np.random.permutation(x.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_x = x[permutation][:batch_size]
    shuffled_y = y[permutation][:batch_size]
    return shuffled_x, shuffled_y

def split_by_class(y_data) :
    img_index = {}
    labels = set(y_data)
    for i,y in enumerate(y_data) :
        if y not in img_index.keys() :
            img_index[y] = [i]
        else :
            img_index[y].append(i)
    return img_index



def random_scaling(img):
    rows, cols, _ = img.shape
    # transform limits
    px = np.random.randint(-2, 2)
    # ending locations
    pts1 = np.float32([[px, px], [rows - px, px], [px, cols - px], [rows - px, cols - px]])
    # starting locations (4 corners)
    pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (rows, cols))
    dst = dst[:, :,np.newaxis]
    return dst


# translation
def random_translate(img):
    rows, cols, _ = img.shape
    # allow translation up to px pixels in x and y directions
    px = 2
    dx, dy = np.random.randint(-px, px, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    dst = dst[:, :,np.newaxis]
    return dst

# rotation
def rotate(img, theta=18):
    r, c = img.shape[:-1]
    new_img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
    rows, cols = new_img.shape[:-1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    new_img = cv2.warpAffine(new_img, M, (cols, rows))
    r0, c0 = round((rows - r) / 2), round((cols - c) / 2)
    return new_img[r0:r0 + r, c0:c0 + c]

def random_transform(img) :
    new_img = img
    transform_list = [scale, translate, rotate]
    random.shuffle(transform_list)
    for transform in transform_list :
        new_img = transform(new_img)
    return new_img

def augment_data(X_data, y_data, n=300):
    X_data_fake, y_data_fake = [],[]
    img_index = split_by_class(y_data)
    for label in range(n_classes):
        n_fake = n-len(img_index[label])
        if n_fake <= 0 : continue
        for i in range(n_fake):
            i_img = random.choice(img_index[label])
            img = X_data[i_img]
            X_data_fake.append(random_transform(img))
            y_data_fake.append(label)
    return np.array(X_data_fake),np.array(y_data_fake)

def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst


def random_warp(img):
    rows, cols, _ = img.shape
    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06  # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06
    # 3 starting points for transform, 1/4 way from edges
    x1 = cols / 4
    x2 = 3 * cols / 4
    y1 = rows / 4
    y2 = 3 * rows / 4
    pts1 = np.float32([[y1, x1],
                       [y2, x1],
                       [y1, x2]])
    pts2 = np.float32([[y1 + rndy[0], x1 + rndx[0]],
                       [y2 + rndy[1], x1 + rndx[1]],
                       [y1 + rndy[2], x2 + rndx[2]]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    dst = dst[:, :,np.newaxis]
    return dst

X_train=np.sum(X_train/3, axis=3, keepdims=True)
X_train=normalize(X_train)
X_test=np.sum(X_test/3, axis=3, keepdims=True)
X_test=normalize(X_test)
X_valid=np.sum(X_valid/3, axis=3, keepdims=True)
X_valid=normalize(X_valid)

print(X_train.shape,X_test.shape,X_valid.shape)
print(y_train.shape,y_test.shape,y_valid.shape)

import os
if os.path.exists('X_train_aug.pickle'):
    pass
else:
    input_indices = []
    output_indices = []
    for class_n in range(n_classes):
        class_indices = np.where(y_train == class_n)
        n_samples = len(class_indices[0])
        if n_samples < 500:
            for i in range(500 - n_samples):
                input_indices.append(class_indices[0][i % n_samples])
                output_indices.append(X_train.shape[0])
                new_img = X_train[class_indices[0][i % n_samples]]
                new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
                X_train = np.concatenate((X_train, [new_img]), axis=0)
                y_train = np.concatenate((y_train, [class_n]), axis=0)
    print("augmentation done", X_train.shape, y_train.shape)

    with open('X_train_aug.p', 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('y_train_aug.p', 'wb') as handle:
        pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)




#X_train, y_train = augment_data(X_train, y_train, n=500)


#print("The below picture shows after normalization:",image_id)
#plt.subplot(122)
#plt.imshow(X_train[index])
#plt.show()

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
batch_size=128
epochs=50
def CNN(input_x,keep_prob):
    mu=0
    sigma=0.1
    print("CNN model building..........")
    #output of conv1 will be [30,30,16], output of pool1 will be [15,15,16]
    with tf.name_scope('conv0'):
        con_w0=tf.Variable(tf.truncated_normal(shape=(3,3,1,16),mean=mu,stddev=sigma))
        conv0=tf.nn.conv2d(input_x,filter=con_w0,strides=[1,1,1,1],padding='VALID')
        conv0=tf.nn.bias_add(conv0,tf.Variable(tf.zeros(16)))
        conv0 = tf.nn.relu(conv0)
        pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # output of conv1 will be [13,13,64], output of pool1 will be [6,6,64]
    with tf.name_scope('conv1'):
        con_w1=tf.Variable(tf.truncated_normal(shape=(3,3,16,64),mean=mu,stddev=sigma))
        conv1=tf.nn.conv2d(pool0,filter=con_w1,strides=[1,1,1,1],padding='VALID')
        conv1=tf.nn.bias_add(conv1,tf.Variable(tf.zeros(64)))
        conv1=tf.nn.relu(conv1)
        pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        #pool1 = tf.nn.dropout(pool1, keep_prob)

    # output of conv2 will be [4,4,128], output of pool2 will be [2,2,128]
    with tf.name_scope('conv2'):
        con_w2=tf.Variable(tf.truncated_normal(shape=(3,3,64,128),mean=mu,stddev=sigma))
        conv2=tf.nn.conv2d(pool1,filter=con_w2,strides=[1,1,1,1],padding='VALID')
        conv2=tf.nn.bias_add(conv2,tf.Variable(tf.zeros(128)))
        conv2=tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        pool2=tf.nn.dropout(pool2, keep_prob)


    #flatten size will be 512
    pool2_flatten=flatten(pool2)

    with tf.name_scope('dense2'):
        den_w2=tf.Variable(tf.truncated_normal(shape=(512,84),mean=mu,stddev=sigma))
        dense2=tf.add(tf.matmul(pool2_flatten,den_w2),tf.zeros(84))
        dense2=tf.nn.relu(dense2)

    with tf.name_scope('drop_out2'):
        dense2=tf.nn.dropout(dense2,keep_prob=keep_prob)

    #output shape will be 43 classes
    with tf.name_scope('output'):
        den_w3=tf.Variable(tf.truncated_normal((84,n_classes),mean=mu,stddev=sigma))
        logits=tf.add(tf.matmul(dense2,den_w3),tf.zeros(n_classes))
    return logits



input_x=tf.placeholder(tf.float32,shape=[None,32,32,1])
input_y=tf.placeholder(tf.int32,shape=[None,])
onehot_y = tf.one_hot(input_y, depth=n_classes)
keep_prob=tf.placeholder(tf.float32)

logits=CNN(input_x,keep_prob)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_y))
op=tf.train.AdamOptimizer(learning_rate=10e-4).minimize(loss)
prediction=tf.argmax(logits,1)
correct_prediction=tf.equal(prediction,tf.argmax(onehot_y,1))
accuracy_=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver=tf.train.Saver()

def evaluate(X_data,y_data):
    n_X=len(X_data)
    total_accuracy=0
    sess=tf.get_default_session()
    for offset in range(0,n_X,batch_size):
        batch_x,batch_y=X_data[offset:offset+batch_size],y_data[offset:offset+batch_size]
        accuracy=sess.run(accuracy_,feed_dict={input_x: batch_x,input_y: batch_y,keep_prob:1.0})
        total_accuracy+=(accuracy*len(batch_x))
    return total_accuracy/n_X


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_X = len(X_train)
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0,n_X,batch_size):
            end=offset+batch_size
            batch_x,batch_y=X_train[offset:end],y_train[offset:end]
            #batch_x_aug,batch_y_aug=augmentation(batch_x, batch_y)
            _,loss_train,accuracy_train=sess.run([op, loss,accuracy_],feed_dict={input_x:batch_x, input_y: batch_y,keep_prob:0.5})
        print('Epoch',i+1,' Training loss:',loss_train,' Training accuracy:',accuracy_train)

        validation_accuracy=evaluate(X_valid,y_valid)
        print("Validation accuracy:",validation_accuracy)
    saver.save(sess, './lenet')

with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('.'))
    test_accuracy=evaluate(X_test,y_test)
    print("Final Test accoracy={:.3f}".format(test_accuracy))
### Train your model here.

Test_web=[]
for pic in os.listdir(r'./Test_web'):
    pic=cv2.imread('./Test_web/'+pic)
    pic2=cv2.resize(pic,(32,32))
    Test_web.append(pic)

Test_web=np.sum(Test_web/3, axis=3, keepdims=True)
Test_web=normalize(Test_web)
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('.'))
    prediction_web=sess.run(prediction,feed_dict={input_x: Test_web,keep_prob:1.0})
    print(prediction_web)




### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.



### Load the images and plot them here.
### Feel free to use as many code cells as needed.




### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.





### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.





### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Feel free to use as many code cells as needed.




def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")