from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


# There are roughly 1.4 million matrix operations, 1.1 million if only considering the convolutional
# operations.

def conn_net():
    l = [   make_connected_layer(3072, 320),
            make_activation_layer(LRELU),
            make_connected_layer(320, 256),
            make_activation_layer(LRELU),
            make_connected_layer(256, 128),
            make_activation_layer(LRELU),
            make_connected_layer(128, 128),
            make_activation_layer(LRELU),
            make_connected_layer(128, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

# This conn_net also has roughly as many operations.

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer: The fully connected network had a training accuracy of 55% and test accuracy of 50%. On the other hand,
# the convnet had an training accuracy of 69% and a test accuracy of 64%. These results are likely a product of the fact
# that fully connected networks do a good job of connecting all pixels whereas convolutional networks find relationships
# between pixels nearby each other. This makes convolutional networks particularly good at classifying images,
# as images are identifiable by the features located nearby spatially, rather than from using every connection of pixels.

