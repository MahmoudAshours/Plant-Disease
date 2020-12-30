import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops
import tensorflow as tf

# Listing the training directory & testing directory which contain images for learning
TRAIN_DIR = 'train/train'
TEST_DIR = 'test/test'
# Each image size can't be more than 50 px to use less memory

IMG_SIZE = 50

# In machine learning and statistics,
# the learning rate is a tuning parameter
# in an optimization algorithm that determines the step size at each iteration while moving
# toward a minimum of a loss function.Since it influences to what extent newly acquired information overrides
# old information, it metaphorically represents the speed at which a machine learning model "learns".
# In the adaptive control literature, the learning rate is commonly referred to as gain.
LR = 1e-3

# Giving a name to the model to save it use it later.
MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

# Configuring Cuda , to give this project enough ram & drivers to complete it's work
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# Used to figure the label of the image whether it's healthy , bacterial , viral or lateblight annotated with the
# first letter of each word
def label_img(img):
    word_label = img[0]

    if word_label == 'h':
        return [1, 0, 0, 0]

    elif word_label == 'b':
        return [0, 1, 0, 0]
    elif word_label == 'v':
        return [0, 0, 1, 0]
    elif word_label == 'l':
        return [0, 0, 0, 1]


# Preparing training data using numpy library
# Steps of preparing data :
# 1) Finding the training data directory.
# 2) Reading each image in the directory & resizing it to 50 x 50 px .
# 3) Adding it to training_data array.
# 4) Saving it using numpy to train_data.npy in order to use the data of the training images.
# An NPY file is a NumPy array file created by the Python software package with the NumPy library installed
# . It contains an array saved in the NumPy (NPY) file format. NPY files store all the information
# required to reconstruct an array on any computer, which includes data type and shape information.
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# Preparing testing data using numpy library
# Steps of preparing data :
# 1) Finding the testing data directory.
# 2) Reading each image in the directory & resizing it to 50 x 50 px .
# 3) Adding it to testing_data array.
# 4) Saving it using numpy to test_data.npy in order to use the data of the testing images.
# An NPY file is a NumPy array file created by the Python software package with the NumPy library installed
# . It contains an array saved in the NumPy (NPY) file format. NPY files store all the information
# required to reconstruct an array on any computer, which includes data type and shape information.
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# If you have already created the dataset:
# train_data = np.load('train_data.npy')
train_data = create_train_data()

# Clears the default graph stack and resets the global default graph
# , which is more like  you clean these nodes in the default graph.
ops.reset_default_graph()

# Start of learning process
# Before learning , What is CNN or conv_net?
"""
Think about Facebook a few years ago, after you uploaded a picture to your profile,
you were asked to add a name to the face on the picture manually. 
Nowadays, Facebook uses convnet to tag your friend in the picture automatically.
A convolutional neural network is not very difficult to understand. 
An input image is processed during the convolution phase and later attributed a label.
The most critical component in the model is the convolutional layer. This part aims at reducing the size of the image 
for faster computations of the weights and improve its generalization.
Convolutional Neural network compiles different layers before making a prediction.
"""
# First of all, an image is pushed to the network; this is called the input image
conv_net = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

"""
At the end of the convolution operation, the output is subject to an activation function to allow non-linearity. 
The usual activation function for convnet is the Relu. All the pixel with a negative value will be replaced by zero.
"""

conv_net = conv_2d(conv_net, 32, 3, activation='relu')

"""
The purpose of the pooling is to reduce the dimensionality of the input image.
The steps are done to reduce the computational complexity of the operation.
 By diminishing the dimensionality, the network has lower weights to compute, so it prevents overfitting.
"""
conv_net = max_pool_2d(conv_net, 3)

conv_net = conv_2d(conv_net, 64, 3, activation='relu')
conv_net = max_pool_2d(conv_net, 3)

conv_net = conv_2d(conv_net, 128, 3, activation='relu')
conv_net = max_pool_2d(conv_net, 3)

conv_net = conv_2d(conv_net, 32, 3, activation='relu')
conv_net = max_pool_2d(conv_net, 3)

conv_net = conv_2d(conv_net, 64, 3, activation='relu')
conv_net = max_pool_2d(conv_net, 3)

conv_net = fully_connected(conv_net, 1024, activation='relu')
conv_net = dropout(conv_net, 0.8)

"""
The last step consists of building a traditional artificial neural network as you did above.
You connect all neurons from the previous layer to the next layer.
You use a softmax activation function to classify the number on the input image.
"""

conv_net = fully_connected(conv_net, 4, activation='softmax')
conv_net = regression(conv_net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(conv_net, tensorboard_dir='log')

# Get bunch of train_data and dividing it with validation data
train = train_data[:-500]
test = train_data[-500:]

# Reshaping each image in training data
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = [i[1] for i in train]

# Reshaping each image in testing data

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_y = [i[1] for i in test]

# Start process learning ( Fitting all variables)
model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

# Saving the model to use it in UI
model.save(MODEL_NAME)
