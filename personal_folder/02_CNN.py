#%%
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import datetime
import math
import random
#%%
tf.__version__
#%%
#### Configuration of Neutral Network
# Convolution Layer 1
filter_size1 = 5  # Convolution filters are 5x5 pixels
num_filters1 = 16  # There are 16 filters

# Convolution Layer 2
filter_size2 = 5 # Convolution filters are 5x5 pixels
num_filters2 = 36 # There are 36 filters

# Fully connected layer
fc_size = 128  # Number of neurons in fully-connected layer
#%%
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', reshape=True, one_hot=False)
#%%
train_images = mnist.train.images
train_cls_true = mnist.train.labels
test_images = mnist.test.images
test_cls_true = mnist.test.labels
vali_images = mnist.validation.images
vali_cls_true = mnist.validation.labels

#%%
train_y_true = tf.one_hot(train_cls_true, depth=num_classes)
test_y_true = tf.one_hot(test_cls_true, depth=num_classes)
vali_y_true = tf.one_hot(vali_cls_true, depth=num_classes)
with tf.Session() as sess:
    train_y_true, test_y_true, vali_y_true = sess.run([train_y_true, test_y_true, vali_y_true])

#%%
print('Size of:')
print('{0:<20}{1}'.format('- Training-set:', train_images.shape[0]))
print('{0:<20}{1}'.format('- Validation-set:', vali_images.shape[0]))
print('{0:<20}{1}'.format('- Test-set:', test_images.shape[0]))
#%%
# The number of pixels in each dimension of an image.
img_size = 28
# The images are stored in one-dimensional arrays of this length
img_size_flat = train_images.shape[1]
# Tuple with height and width of images used to reshape arrays
img_shape = (28, 28)
# Number of clasess, one class for each of 10 digits
num_classes = 10
# Number of colour channels for the images: 1 channel for gray-scale
num_channels = 1
#%%
# Function used to plot 9 images in a 3x3 grid, and writing the true and 
# predicted classes below each image
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Show the true and predicted classes
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

#%%
# Get the first images from the test
images = train_images[:9]
cls_true = train_cls_true[:9]
plot_images(images, cls_true)

#%%
# Define functions for creating new TF variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#%%
# Define a function for creating a new convolutional layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters,
                   use_pooling=True):
    """A function for creating a new convolutional layer

    Params:
        input: the previous layer
        num_input_channels: number of channels in the previous layer
        num_filters: number of filters
        use_pooling: use 2x2 max-pooling
    return: 4-dim tensor with the following dimensions:
                1. Image number
                2. Y-axis of each image
                3. X-axis of each image
                4. Channels of each image
    """
    # Shape of the filter-weihts for the convolution
    # This format is determined by the TF API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape
    weights = new_weights(shape)

    # Create new biases, one for each filter
    biases = new_biases(length=num_filters)

    # Create the TF operation for convolution.
    # Note the strides are set to 1 in all dimension.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME', which mean that the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input, 
                         filter=weights,
                         strides=[1, 1, 1, 1], 
                         padding='SAME')
    # Add the biases to the results of convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, 
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # Rectified Linear Unit (ReLu).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that the ReLU is normally executed before the pooling,
    # but since relu(max_pool(x) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later
    return layer, weights

#%%
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape = [num_images, img_height, img_weight, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TF to calculate this
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features]
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in the dimension is calculated
    # so the total size of the tensor is unchanaged from the reshaping.
    layer_flat = tf.reshape(layer, shape=[-1, num_features])

    # The shape of the flatten layer is now:
    # [num_images, img_height*img_width*num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features
#%%
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    """This function creates a new fully-connected layer in
    the computational graph for TF.

    Params:
        input: the previous layer. It is assumed that the input is a 2-D
               tensor of shape [num_images, num_inputs]
        num_inputs: the number of inputs from prev. layer.
        num_outputs: the number of outputs.
        use_relu: use Rectified Linear Unit (ReLU)?
    Return: a 2-D tensor of shape [num_images, num_output]
    """
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

#%%
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
# Reshape the input x to [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

#%%
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)



#%%
layer_conv1

#%%
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

#%%
layer_conv2

#%%
layer_flat, num_features = flatten_layer(layer_conv2)

#%%
layer_flat

#%%
layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, use_relu=True)

#%%
layer_fc1

#%%
layer_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, False)

#%%
layer_fc2

#%%
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

#%%
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                        labels=y_true)

#%%
cost = tf.reduce_mean(cross_entropy)

#%%
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#%%
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%%
train_batch_size = 64

#%%
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    num_train_images = train_images.shape[0]
    idx = range(num_train_images)

    for i in range(total_iterations, total_iterations + num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images
        batch_idx = random.sample(idx, train_batch_size)
        x_batch = train_images[batch_idx, :]
        y_true_batch = train_y_true[batch_idx, :]

        feed_dict = {x: x_batch, y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict)

        # Print status every 100 iterations
        if i % 100 == 0:
            # Calculate the accuracy on the training set
            acc = sess.run(accuracy, feed_dict=feed_dict)

            # Message for printing
            msg = "Optimisation Iteration: {0: >6}, Training Accurancy: {1: >6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()
    time_diff = end_time - start_time

    # Print the time-usage.
    print("Time usage: ", time_diff)

#%%
def plot_confusion_matrix(cls_pred, cls_true):
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    # Print the confusion table as test
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = range(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#%%
def plot_example_errors(images, cls_true, cls_pred, correct_pred):
    incorrect = [correct_pred==False]
    incorrect_images = images[incorrect]
    labels_true = cls_true[incorrect]
    labels_pred = cls_pred[incorrect]
    plot_images(incorrect_images[:9], labels_true[:9], labels_pred[:9])

#%%
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # Number of images in the test-set
    num_test = test_images.shape[0]

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    cls_true = np.zeros(shape=num_test, dtype=np.int)

    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)

        images = test_images[i:j]
        labels = test_cls_true[i:j]

        feed_dict = {x: images}

        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        cls_true[i:j] = labels

        i = j

    correct = (cls_pred==cls_true)
    correct_sum = correct.sum()
    acc = correct_sum/num_test

    # Print the accuracy
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(test_images, cls_true, cls_pred, correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred, cls_true)

#%%
print_test_accuracy()

#%%
optimize(num_iterations=10000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

#%%
