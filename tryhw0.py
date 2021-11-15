from uwnet import *

mnist = 0

# Upon training our own model with some modifications to our hyperparameters, we were able to get mnist
# with a test accuracy > 97%.

# For CIFAR, the model needed to be a lot more complex to achieve a comparable accuracy. If we changed
# the model to be for MNIST, the model decreasing in complexity will impact the CIFAR results far more
# than for MNIST. 

inputs = 784 if mnist else 3072

def softmax_model():
    l = [make_connected_layer(inputs, 10),
        make_activation_layer(SOFTMAX)]
    return make_net(l)

# Designed this for CIFAR
def neural_net():
    l = [   make_connected_layer(inputs, 128),
            make_activation_layer(LRELU),
            make_connected_layer(128, haikyu64),
            make_activation_layer(LRELU),
            make_connected_layer(64, 32),
            make_activation_layer(LRELU),
            make_connected_layer(32, 16),
            make_activation_layer(LRELU),
            make_connected_layer(16, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
if mnist:
    train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
    test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
else:
    train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
    test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 512
iters = 5000
rate = .01
momentum = .9
decay = .01

m = neural_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))
