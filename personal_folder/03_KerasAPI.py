#%%
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

#%%
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import (InputLayer, Input, Reshape,
                                            MaxPooling2D, Conv2D, Dense, Flatten)

#%%
tf.__version__

#%%
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', reshape=True, one_hot=False)

#%%
train_images = mnist.train.images
train_classes = mnist.train.labels
test_images = mnist.test.images
test_classes = mnist.test.labels
validate_images = mnist.validation.images
validate_classes = mnist.validation.labels

#%%
train_y_true = tf.one_hot(train_classes, depth=10)
test_y_true = tf.one_hot(test_classes, depth=10)
validate_y_true = tf.one_hot(validate_classes, depth=10)
with tf.Session() as sess:
    train_y_true, test_y_true, validate_y_true \
        = sess.run([train_y_true, test_y_true, validate_y_true])

#%%
print('Size of:')
print('{0:<20} {1}'.format('- Training-set:', train_classes.shape[0]))
print('{0:<20} {1}'.format('- Validation-set:', validate_images.shape[0]))
print('{0:<20} {1}'.format('- Test-set:', test_images.shape[0]))

#%%
# The images are stored in one-dimensional arrays of this length
image_size_flat = train_images.shape[1]
# Tuple with height and width of images used to reshape arrays
image_shape = (28, 28)
# Number of clasess, one class for each of 10 digits
num_classes = 10
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
# Tuple with height and width of images used to reshape arrays.
# This is used for reshaping in Keras.
image_shape_full = (28, 28, 1)

print(image_size_flat, image_shape, num_classes, num_channels, image_shape_full)

#%%
# Function used to plot 9 images in a 3x3 grid, and writing the true and 
# predicted classes below each image
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Show the true and predicted classes
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(image_shape), cmap='binary')

        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

#%%
images = train_images[:9]
classes_true = train_classes[:9]
plot_images(images, classes_true)

#%%
def plot_example_errors(images, cls_true, cls_pred, correct_pred):
    incorrect = [correct_pred==False]
    incorrect_images = images[incorrect]
    labels_true = cls_true[incorrect]
    labels_pred = cls_pred[incorrect]
    plot_images(incorrect_images[:9], labels_true[:9], labels_pred[:9])

#%%
# Sequential model using Keras API
# Start construction of the Keras Sequential model.
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
model.add(InputLayer(input_shape=(image_size_flat,)))

# The input is a flattened array with 784 elements
# but the convolutional layers expect images with shape (28, 28, 1)
model.add(Reshape(image_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

# First fully-connected /dense layer with ReLU-activation.
model.add(Dense(128, activation='relu'))

# Last fully-connected /dense layer with softmax-activation
# for use in classification
model.add(Dense(num_classes, activation='softmax'))


#%%
# Model compilation
from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr=1e-3)

#%%
# Add a loss-function, optimizer and performance metrics
# which is called "compile" in Keras
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%
model.fit(x=train_images, y=train_y_true, epochs=1, batch_size=128)

#%%
model.evaluate(x=test_images, y=test_y_true)

#%%
images = test_images
cls_true = test_classes

#%%
y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)

#%%
correct_pred = (cls_true==cls_pred)
plot_example_errors(images, cls_true, cls_pred, correct_pred)
#plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred)

#%%
#### Functional Model ####
# Create an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size
inputs = Input(shape=(image_size_flat, ))

# Variable used for building the Neural Network.
net = inputs

# The input is an image as a flattened array with 784 elements.
# But the convolutional layers expect images with shape (28, 28, 1)
net = Reshape(image_shape_full)(net)

# First convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
             activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Flatten the output of the conv-layer from 4-dim to 2-dim.
net = Flatten()(net)

# First fully-connected / dense layer with ReLU-activation
net = Dense(units=128, activation='relu')(net)

# Last fully-connected / dense layer with softmax-activation.
net = Dense(units=num_classes, activation='softmax')(net)

# Output of the Neural Network.
outputs = net

#%%
# Create a new instance of the Keras Functional Model
model12 = Model(inputs=inputs, outputs=outputs)

#%%
model12.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

#%%
model12.fit(x=train_images, y=train_y_true, epochs=3, batch_size=256)

#%%
result = model12.evaluate(x=test_images, y=test_y_true)

#%%
for name, value in zip(model12.metrics_names, result):
    print(name, value)

#%%
y_pred = model12.predict(x=validate_images)
cls_pred = np.argmax(y_pred, axis=1)

#%%
plot_images(validate_images[:9], validate_classes[:9], cls_pred[:9])

#%%
correct_pred = (cls_pred==validate_classes)
plot_example_errors(validate_images, validate_classes, cls_pred, correct_pred)

#%%
model12.save('model.keras')

#%%
del model12

#%%
from tensorflow.python.keras.models import load_model

#%%
model13 = load_model('model.keras')

#%%
y_pred = model13.predict(x=validate_images)

#%%
cls_pred = np.argmax(y_pred, axis=1)
plot_images(validate_images, validate_classes, cls_pred)