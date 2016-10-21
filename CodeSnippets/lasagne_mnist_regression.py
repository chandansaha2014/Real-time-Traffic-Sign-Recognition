from __future__ import print_function

import sys
import os
import time
import pickle

import numpy as np
import theano
import theano.tensor as T

import lasagne
import nolearn
import nolearn.lasagne


def load_dataset_1():
    # load the data
    dataset_path = "D:\\_Dataset\\MNIST\\mnist.pkl"
    # Load the dataset
    data = pickle.load(open(dataset_path, 'rb'))

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))

    # The targets are int64, we cast them to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset_2():
    # load the data
    dataset_path = "D:\\_Dataset\\MNIST\\mnist.pkl"
    # Load the dataset
    data = pickle.load(open(dataset_path, 'rb'))

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))

    y_train = y_train.reshape((-1, 1))
    y_val = y_val.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # The targets are int64, we cast them to float and normalize them
    # to be in range from [0,1] or from [-1, 1]
    y_train = ((y_train.astype(np.float32) * 2) - 9.0) / 9.0
    y_val = ((y_val.astype(np.float32) * 2) - 9.0) / 9.0
    y_test = ((y_test.astype(np.float32) * 2) - 9.0) / 9.0

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def float32(k):
    return np.cast['float32'](k)


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model build in Lasagne.
def build_cnn_1(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    layer = lasagne.layers.Conv2DLayer(
        layer, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    layer = lasagne.layers.MaxPool2DLayer(layer, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    layer = lasagne.layers.Conv2DLayer(
        layer, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    layer = lasagne.layers.MaxPool2DLayer(layer, pool_size=(2, 2))

    # A fully-connected layer of 500 units with 50% dropout on its inputs:
    layer = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(layer, p=.5),
        num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 1-unit output regression layer with no dropout on its inputs:
    layer = lasagne.layers.DenseLayer(layer, num_units=1, nonlinearity=None)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    # layer = lasagne.layers.DenseLayer(lasagne.layers.dropout(layer, p=.5), num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    return layer


def build_cnn_2():
    net = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('conv1', lasagne.layers.Conv2DLayer),
            ('pool1', lasagne.layers.MaxPool2DLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('conv2', lasagne.layers.Conv2DLayer),
            ('pool2', lasagne.layers.MaxPool2DLayer),
            ('dropout2', lasagne.layers.DropoutLayer),
            ('hidden3', lasagne.layers.DenseLayer),
            ('dropout3', lasagne.layers.DropoutLayer),
            ('hidden4', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=(None, 1, 28, 28),
        conv1_num_filters=20, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=60, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        hidden3_num_units=1000,
        dropout3_p=0.5,
        hidden4_num_units=1000,
        output_num_units=1, output_nonlinearity=None,
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=500),
        eval_size=0.0,
        regression=True,
        max_epochs=1,
        verbose=1,
    )
    return net


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def train_1():
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_1()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.fvector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn_1(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem in regression fashion, it is L-2 norm loss):
    prediction = lasagne.layers.get_output(layer_or_layers=network, regression=True)
    loss = lasagne.objectives.squared_error(prediction.reshape((-1,)), target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction.reshape((-1,)), target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)

    # prediction
    predict_fn = theano.function([input_var, target_var], test_prediction)

    # Finally, launch the training loop.
    print("Starting training...")
    num_epochs = 5
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            print("... epoch: %d/%d, batch: %d" % (epoch + 1, num_epochs, train_batches))
            break

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            predict = predict_fn(inputs, targets)
            val_err += err
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_err / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_err / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))


def train_2():
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_2()

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    # network = build_cnn_1(input_var)
    network = build_cnn_2()
    network.fit(X_train, y_train)

    model_path = "D:\\_Dataset\\GTSRB\\las_model_mnist_28.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(network, f, -1)

    print("... finish training the model")


def train_3():
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_2()

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    # network = build_cnn_1(input_var)
    network = build_cnn_2()

    # Finally, launch the training loop.
    print("Starting training...")
    num_epochs = 1
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500):
            inputs, targets = batch
            network.fit(inputs, targets)
            # no need to do predictions for the mini-batch
            # if you set verbous option in the network, it will
            # print the RMSE after fitting each mini-batch
            # pred = network.predict(inputs).reshape(-1, )
            # pred -= np.min(pred)
            # pred *= 9.0 / (np.max(pred))
            # targets = targets.reshape(-1, )
            # targets = (targets + 1) * 9 / 2
            # error = np.mean(np.abs(pred - targets)) * 100 / 9.0
            # train_err += error
            # train_batches += 1
            # print("... epoch: %d/%d, mini-batch: %d, error: %f" % (epoch + 1, num_epochs, train_batches, error))

    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    model_path = "D:\\_Dataset\\GTSRB\\las_model_mnist_28_2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(network, f, -1)

    print("... finish training the model")


def test_2():
    dataset_path = "D:\\_Dataset\\MNIST\\mnist.pkl"
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    x_train, y_train = data[0]
    x_val, y_val = data[1]
    x_test, y_test = data[2]

    x_train = x_train.reshape((-1, 1, 28, 28))
    x_val = x_val.reshape((-1, 1, 28, 28))
    x_test = x_test.reshape((-1, 1, 28, 28))

    del data

    # load model and do predictions
    model_path = "D:\\_Dataset\\GTSRB\\las_model_mnist_28.pkl"
    with open(model_path, "rb") as f:
        network = pickle.load(f)
    y_pred = network.predict(x_val).reshape(-1, )

    # this is an optional step to re-adjust range of the predictions
    # the alternative is to just re-scale like this: y_pred *= 9.0
    y_pred -= np.min(y_pred)
    y_pred *= 9.0 / (np.max(y_pred))

    # convert to int to be compared with the target
    y_pred_int = np.rint(y_pred).astype(int)

    # this is if we calculate the error as a classification problem
    error = np.mean(np.not_equal(y_pred_int, y_val))
    print("... classification error: %f" % (error * 100))

    # but since we trained regression problem, then calculate the error
    # as regression problem, then error is RMSE
    error = np.mean(np.abs(y_pred - y_val)) * 100 / 9.0
    print("... regression error: %f" % (error))
    error = np.sqrt(np.mean((y_pred - y_val) ** 2))
    print("... rmse error: %f" % (error))

    aaa = 10
