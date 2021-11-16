from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 2000
rate = .1
momentum = .9
decay = .005

m = conv_net()
print("training...")
for i in range(5):
    train_image_classifier(m, train, batch, iters, rate, momentum, decay)
    rate /= 10
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# Without batch normalization, the testing accuracy we achieved was 53.3%. With batch normalization, we achieved a test accuracy of 61.3%. With batch normalization, it achieved convergence
# a lot faster than without. With a higher magnitude, the accuracy decreased to 57.6%, even though it was a higher accuracy than without normalization. With annealing, the test accuracy I achieved was 64.0% with 2000 iterations, 
# starting at 0.1 and dividing by 10 each time for 5 times. 
