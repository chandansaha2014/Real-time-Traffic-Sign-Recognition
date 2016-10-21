"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling by max.
 - Digit classification is implemented with a logistic regression rather than an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

import os
import sys
import time
import cv2
import PIL
import PIL.Image

import numpy
import numpy.random

import theano
import theano.tensor as T

import lasagne
import lasagne.layers
import nolearn
import nolearn.lasagne
import nolearn.lasagne.handlers

import pickle
import CNN
import CNN.svm
import CNN.logit
import CNN.utils
import CNN.mlp
import CNN.conv
from CNN.mlp import HiddenLayer
import CNN.enums

# region Train Class Classifier

def train_shallow(dataset_path, model_path='', img_dim=28, learning_rate=0.1, n_epochs=200, kernel_dim=(5, 5), nkerns=(20, 50),
                  mlp_layers=(500, 10), batch_size=500, pool_size=(2, 2)):
    """ Demonstrates cnn on the given dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer1 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = CNN.logit.LogisticRegression(input=layer2.output, n_in=mlp_layers[0], n_out=mlp_layers[1])

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch
    print("... validation freq: %d" % validation_frequency)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            cost_ij = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def train_deep(dataset_path, model_path='', img_dim=80, learning_rate=0.1, n_epochs=200, kernel_dim=(9, 7, 4), nkerns=(10, 58, 360),
               mlp_layers=(500, 17), batch_size=100, pool_size=(2, 2)):
    """ Demonstrates cnn on the given dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    rng = numpy.random.RandomState(23455)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer1 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer2_kernel_dim = kernel_dim[2]
    layer2 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        poolsize=pool_size
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)  # = 4 in case of mnist
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * layer3_img_dim * layer3_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = CNN.logit.LogisticRegression(input=layer3.output, n_in=mlp_layers[0], n_out=mlp_layers[1])

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    print("... validation freq: %d" % validation_frequency)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            cost_ij = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer4.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer4.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def train_deep_lasagne(dataset_path, model_path='', img_dim=80, learning_rates=(0.02, 0.005), momentums=(0.9, 0.95),
                       n_epochs=100, kernel_dim=(13, 5, 4), nkerns=(10, 50, 200),
                       mlp_layers=(500, 200, 12), batch_size=400, pool_size=(2, 2)):
    """
    Train classification model using nolearn lasagne
    """

    # load the data and concatenate all subsets in one set
    # as the nolearn will use them to train and validate
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    data_x = numpy.concatenate((numpy.concatenate((dataset[0][0], dataset[1][0])), dataset[2][0]))
    data_y = numpy.concatenate((numpy.concatenate((dataset[0][1], dataset[1][1])), dataset[2][1]))
    data_x = data_x.reshape(data_x.shape[0], 1, img_dim, img_dim)
    data_x = data_x.astype("float32")

    eval_split = 0.2
    n_minibatches = 100
    n_total = len(data_y)
    n_train = int(n_total * (1 - eval_split))
    batch_size = int(n_train / n_minibatches)

    nn_recognition = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('conv1', lasagne.layers.Conv2DLayer),
            ('pool1', lasagne.layers.MaxPool2DLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('conv2', lasagne.layers.Conv2DLayer),
            ('pool2', lasagne.layers.MaxPool2DLayer),
            ('dropout2', lasagne.layers.DropoutLayer),
            ('conv3', lasagne.layers.Conv2DLayer),
            ('pool3', lasagne.layers.MaxPool2DLayer),
            ('dropout3', lasagne.layers.DropoutLayer),
            ('hidden4', lasagne.layers.DenseLayer),
            ('dropout4', lasagne.layers.DropoutLayer),
            ('hidden5', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=(None, 1, img_dim, img_dim),
        conv1_num_filters=nkerns[0], conv1_filter_size=(kernel_dim[0], kernel_dim[0]), pool1_pool_size=pool_size,
        dropout1_p=0.1,
        conv2_num_filters=nkerns[1], conv2_filter_size=(kernel_dim[1], kernel_dim[1]), pool2_pool_size=pool_size,
        dropout2_p=0.2,
        conv3_num_filters=nkerns[2], conv3_filter_size=(kernel_dim[2], kernel_dim[2]), pool3_pool_size=pool_size,
        dropout3_p=0.3,
        hidden4_num_units=mlp_layers[0],
        dropout4_p=0.5,
        hidden5_num_units=mlp_layers[1],
        output_num_units=mlp_layers[2],
        output_nonlinearity=lasagne.nonlinearities.softmax,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rates[0])),
        update_momentum=theano.shared(CNN.utils.float32(momentums[0])),
        train_split=nolearn.lasagne.TrainSplit(eval_size=eval_split),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        max_epochs=n_epochs,
        verbose=1,
        on_epoch_finished=[
            CNN.utils.AdjustVariable('update_learning_rate', start=learning_rates[0], stop=learning_rates[1]),
            CNN.utils.AdjustVariable('update_momentum', start=momentums[0], stop=momentums[1]),
            CNN.utils.EarlyStopping(patience=200),
        ])

    ##############################
    # Train the model            #
    ##############################

    # Finally, launch the training loop.
    print("... training the recognition model")
    print("... in total, %d samples in training" % (n_train))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))
    start_time = time.clock()
    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    nn_recognition.fit(data_x, data_y)

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    if len(model_path) > 0:
        with open(model_path, "wb") as f:
            pickle.dump(nn_recognition, f, -1)


def train_linear_classifier(dataset_path, model_path='', img_dim=28, learning_rate=0.1, n_epochs=200, kernel_dim=(5, 5), nkerns=(20, 50),
                            mlp_layers=(500, 10), batch_size=50, pool_size=(2, 2)):
    """ Demonstrates cnn on the given dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer1 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist
    layer2 = CNN.mlp.HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = CNN.logit.LinearRegression(input=layer2.output, n_in=mlp_layers[0], n_out=mlp_layers[1])

    # the cost we minimize during training is the NLL of the model
    cost = layer3.cost(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    # updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
    updates = []
    i = 0
    l_rate_slow = 0.01
    l_rate_fast = 0.1
    for param_i, grad_i in zip(params, grads):
        l_rate = l_rate_slow if i == 0 else l_rate_fast
        updates.append((param_i, param_i - l_rate * grad_i))
        i += 1

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch
    print("... validation freq: %d" % validation_frequency)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            cost_ij = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def train_cnn_svm(dataset_path, model_path='', img_dim=28, learning_rate=0.1, n_epochs=200, kernel_dim=(5, 5),
                  nkerns=(20, 50), mlp_layers=(500, 10), batch_size=500, pool_size=(2, 2)):
    """ Demonstrates cnn on the given dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_set_y_h = CNN.svm.y_one_hot(train_set_y.eval(), mlp_layers[1])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    y_h = T.imatrix('y_h')  # for training, the labels are presented as 1D vector of one-hot {-1, 1} labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer1 = CNN.conv.ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    # classify it using SVM instead of logistic regression
    layer3 = CNN.svm.SVMLayer(input=layer2.output, n_in=mlp_layers[0], n_out=mlp_layers[1])

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # the cost we minimize during training is the cost of the svm not the negative-log-liklihood of the logsitics regressor
    cost = layer3.cost(y_h)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y_h: train_set_y_h[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    print("... validation freq: %d" % validation_frequency)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        print("... epoch: %d" % epoch)

        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('... training @ iter = %.0f' % iter)

            # train the minibatch
            cost_ij = train_model(minibatch_index)

            if (iter + 1) == validation_frequency:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print('... epoch %d, minibatch %d/%d, validation error %.2f %%' % (
                    epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(int(n_test_batches))]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of best model %.2f%%') % (
                        epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %.2f%% obtained at iteration %i with test performance %.2f%%' % (
        best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(sys.stderr)

    if len(model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(model_path, 'wb')
    pickle.dump(dataset_path, save_file, -1)
    pickle.dump(img_dim, save_file, -1)
    pickle.dump(kernel_dim, save_file, -1)
    pickle.dump(nkerns, save_file, -1)
    pickle.dump(mlp_layers, save_file, -1)
    pickle.dump(pool_size, save_file, -1)
    pickle.dump(layer0.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer0.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
    pickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
    save_file.close()


def resume_training_lasagne(dataset_path, model_path, learning_rates=(0.05, 0.008), momentums=(0.9, 0.95), n_epochs=50, save_model=True):
    # load the data
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate all subsets in one set as the nolearn will use them to train and validate
    train_x = numpy.concatenate((dataset[0][0], dataset[1][0]))
    train_y = numpy.concatenate((dataset[0][1], dataset[1][1]))
    del dataset

    img_dim = 28
    n_batches = 10
    batch_size = int(train_y.shape[0] / n_batches)

    # reshape images so it can be ready for convolution
    train_x = train_x.reshape((train_x.shape[0], 1, img_dim, img_dim))

    # load the regression model
    print('... loading model')
    with open(model_path, 'rb') as f:
        net_cnn = pickle.load(f)

    net_cnn.max_epochs = n_epochs
    n_steps = len(net_cnn.train_history_) + n_epochs
    net_cnn.on_epoch_finished = [
        CNN.utils.AdjustVariable('update_learning_rate', start=learning_rates[0], stop=learning_rates[1], count=n_steps),
        CNN.utils.AdjustVariable('update_momentum', start=momentums[0], stop=momentums[1], count=n_steps),
        CNN.utils.EarlyStopping(patience=200),
        nolearn.lasagne.handlers.PrintLog(),
        nolearn.lasagne.handlers.PrintLayerInfo(),
    ]

    ##############################
    # Train The Regression Model #
    ##############################

    n_minibatches = int(train_y.shape[0] / batch_size)
    print("... training the regression model")
    print("... in total, %d samples in training" % (train_y.shape[0]))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))

    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    start_time = time.clock()
    net_cnn.fit(train_x, train_y)
    end_time = time.clock()

    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    # save the model (optional)
    if save_model:
        with open(model_path, "wb") as f:
            pickle.dump(net_cnn, f, -1)

    dummy_variable = 10


# endregion

#  region Test Classification/Recognition

def classify_imgs_from_files(imgs_pathes, model_path, classifier=CNN.enums.ClassifierType.logit, img_dim=28, is_rgb=False):
    imgs = []
    for path in imgs_pathes:
        img = []
        if is_rgb:
            img = CNN.utils.rgb_to_gs(path)
            img = img.astype(dtype='float64') / 255.0
        else:
            img = PIL.Image.open(path)
            img = numpy.asarray(img, dtype='float64') / 255.0
        imgs.append(img)

    n = imgs.shape[1]
    imgs = numpy.asarray(imgs.shape[0], n * n)
    classify_batch(imgs, model_path, classifier)


def classify_img_from_file(img_path, img_dim, model_path, model_type=CNN.enums.ModelType._01_conv2_mlp2,
                           classifier=CNN.enums.ClassifierType.logit, is_rgb=False):
    # this is how to prepare an image to be used by the CNN model

    img = []
    if is_rgb:
        img = CNN.utils.rgb_to_gs(img_path)
        img = img.astype(dtype='float64') / 255.0
    else:
        img = PIL.Image.open(img_path)
        img = numpy.asarray(img, dtype='float64') / 255.0

    img4D = img.reshape(1, 1, img_dim, img_dim)
    return __classify_img(img4D, model_path, model_type, classifier)


def classify_img_from_dataset(dataset_path, model_path, index, classifier=CNN.enums.ClassifierType.logit, img_dim=28):
    data = pickle.load(open(dataset_path, 'rb'))
    img = data[0][0][index]
    del data
    img4D = img.reshape(1, 1, img_dim, img_dim)

    return __classify_img(img, model_path, classifier)

    # this is if image is loaded from tensor dataset
    # img = test_set_x[index]
    # img = img.eval()
    # img4D = img.reshape(1, 1, img_dim, img_dim)


def classify_batch(batch, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)

    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist

    # layer 0: Conv-Pool
    batch_size = batch.shape[0]
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    batch = batch.reshape(image_shape)

    layer0_input = T.tensor4(name='input')
    layer0_output = CNN.conv.convpool_layer(input=layer0_input, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (batch_size, nkerns[0], layer1_img_dim, layer1_img_dim)
    layer1_output = CNN.conv.convpool_layer(input=layer0_output, W=layer1_W, b=layer1_b,
                                            image_shape=image_shape, filter_shape=filter_shape,
                                            pool_size=pool_size)

    # layer 2: hidden layer
    hidden_n_in = nkerns[1] * layer2_img_dim * layer2_img_dim
    layer1_output_flattened = layer1_output.flatten(2)
    layer2 = CNN.mlp.HiddenLayer(input=layer1_output_flattened, W=layer2_W, b=layer2_b, n_in=hidden_n_in,
                                 n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 3: logit (logistic regression) or SVM
    if classifier == CNN.enums.ClassifierType.logit:
        layer3_y, layer3_y_prob = CNN.logit.logit_layer(input=layer2.output, W=layer3_W, b=layer3_b)
    elif classifier == CNN.enums.ClassifierType.svm:
        layer3_y, layer3_y_prob = CNN.svm.svm_layer(input=layer2.output, W=layer3_W, b=layer3_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    start_time = time.clock()

    # two functions for calculating the result and confidence/probability per class
    f_prob = theano.function([layer0_input], layer3_y_prob)
    f_pred = theano.function([layer3_y_prob], layer3_y)
    c_prob = f_prob(batch)
    c_result = f_pred(c_prob)

    end_time = time.clock()
    duration = end_time - start_time
    return c_result, c_prob, duration


def classify_batch_step_by_step(batch, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)

    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist

    start_time = time.clock()

    # layer 0: Conv-Pool
    batch_size = batch.shape[0]
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    batch = batch.reshape(image_shape)
    (layer0_filters, layer0_output) = CNN.conv.filter_image(img=batch, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (batch_size, nkerns[0], layer1_img_dim, layer1_img_dim)
    (layer1_filters, layer1_output) = CNN.conv.filter_image(img=layer0_filters, W=layer1_W, b=layer1_b,
                                                            image_shape=image_shape, filter_shape=filter_shape,
                                                            pool_size=pool_size)

    # layer 2: hidden layer
    hidden_n_in = nkerns[1] * layer2_img_dim * layer2_img_dim
    layer1_output_flattened = layer1_output.flatten(2)
    hiddenLayer = CNN.mlp.HiddenLayer(input=layer1_output_flattened, W=layer2_W, b=layer2_b, n_in=hidden_n_in,
                                      n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 3: logit (logistic regression) or SVM
    c_result = []
    c_prob = []
    if classifier == CNN.enums.ClassifierType.logit:
        c_result, c_prob = CNN.logit.classify_images(input_flatten=layer1_output_flattened,
                                                     hidden_output=hiddenLayer.output,
                                                     filters=layer1_filters, W=layer3_W,
                                                     b=layer3_b)
    elif classifier == CNN.enums.ClassifierType.svm:
        c_result, c_prob = CNN.svm.classify_images(input_flatten=layer1_output_flattened,
                                                   hidden_output=hiddenLayer.output,
                                                   filters=layer1_filters, W=layer3_W,
                                                   b=layer3_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    end_time = time.clock()
    duration = end_time - start_time
    return c_result, c_prob, duration


def __classify_img(img4D, model_path, model_type=CNN.enums.ModelType._01_conv2_mlp2, classifier=CNN.enums.ClassifierType.logit):
    if model_type == CNN.enums.ModelType._01_conv2_mlp2:
        return __classify_img_shallow_model(img4D, model_path, classifier)
    elif model_type == CNN.enums.ModelType._02_conv3_mlp2:
        return __classify_img_deep_model(img4D, model_path, classifier)
    else:
        raise Exception("Unknown model type")


def __classify_img_shallow_model(img4D, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path, CNN.enums.ModelType._01_conv2_mlp2)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)

    layer0_img_dim = img_dim  # = 28 in case of mnist
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)  # = 4 in case of mnist

    start_time = time.clock()

    # layer 0: Conv-Pool
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (1, 1, layer0_img_dim, layer0_img_dim)
    (layer0_filters, layer0_output) = CNN.conv.filter_image(img=img4D, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (1, nkerns[0], layer1_img_dim, layer1_img_dim)
    (layer1_filters, layer1_output) = CNN.conv.filter_image(img=layer0_filters, W=layer1_W, b=layer1_b,
                                                            image_shape=image_shape, filter_shape=filter_shape,
                                                            pool_size=pool_size)

    # layer 2: hidden layer
    hidden_n_in = nkerns[1] * layer2_img_dim * layer2_img_dim
    layer1_output_flattened = layer1_output.flatten(2)
    hiddenLayer = CNN.mlp.HiddenLayer(input=layer1_output_flattened, W=layer2_W, b=layer2_b, n_in=hidden_n_in,
                                      n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 3: logit (logistic regression) or SVM
    c_result = []
    c_prob = []
    if classifier == CNN.enums.ClassifierType.logit:
        c_result, c_prob = CNN.logit.classify_images(input_flatten=layer1_output_flattened,
                                                     hidden_output=hiddenLayer.output,
                                                     filters=layer1_filters, W=layer3_W,
                                                     b=layer3_b)
    elif classifier == CNN.enums.ClassifierType.svm:
        c_result, c_prob = CNN.svm.classify_images(input_flatten=layer1_output_flattened,
                                                   hidden_output=hiddenLayer.output,
                                                   filters=layer1_filters, W=layer3_W,
                                                   b=layer3_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    end_time = time.clock()

    # that's because we only classified one image
    c_result = c_result[0]
    c_prob = c_prob[0]
    c_duration = end_time - start_time

    # __plot_filters_1(img4D, 1)
    # __plot_filters_1(layer0_filters, 2)
    # __plot_filters_1(layer1_filters, 3)
    # __plot_filters_2(loaded_objects[6], 4)
    # __plot_filters_2(loaded_objects[8], 5)

    print('Classification result: %d in %f sec.' % (c_result, c_duration))
    print('Classification confidence: %s' % (CNN.utils.numpy_to_string(c_prob)))

    return c_result, c_prob, c_duration


def __classify_img_deep_model(img4D, model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path, CNN.enums.ModelType._02_conv3_mlp2)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    mlp_layers = loaded_objects[4]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)
    layer3_W = theano.shared(loaded_objects[12], borrow=True)
    layer3_b = theano.shared(loaded_objects[13], borrow=True)
    layer4_W = theano.shared(loaded_objects[14], borrow=True)
    layer4_b = theano.shared(loaded_objects[15], borrow=True)

    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)

    start_time = time.clock()

    # layer 0: Conv-Pool
    filter_shape = (nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim)
    image_shape = (1, 1, layer0_img_dim, layer0_img_dim)
    (layer0_filters, layer0_output) = CNN.conv.filter_image(img=img4D, W=layer0_W, b=layer0_b, image_shape=image_shape,
                                                            filter_shape=filter_shape, pool_size=pool_size)

    # layer 1: Conv-Pool
    filter_shape = (nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim)
    image_shape = (1, nkerns[0], layer1_img_dim, layer1_img_dim)
    (layer1_filters, layer1_output) = CNN.conv.filter_image(img=layer0_filters, W=layer1_W, b=layer1_b,
                                                            image_shape=image_shape, filter_shape=filter_shape,
                                                            pool_size=pool_size)

    # layer 2: Conv-Pool
    filter_shape = (nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim)
    image_shape = (1, nkerns[1], layer2_img_dim, layer2_img_dim)
    (layer2_filters, layer2_output) = CNN.conv.filter_image(img=layer1_filters, W=layer2_W, b=layer2_b,
                                                            image_shape=image_shape, filter_shape=filter_shape,
                                                            pool_size=pool_size)

    # layer 3: hidden layer
    hidden_n_in = nkerns[1] * layer3_img_dim * layer3_img_dim
    layer2_output_flattened = layer2_output.flatten(2)
    hiddenLayer = CNN.mlp.HiddenLayer(input=layer2_output_flattened, W=layer3_W, b=layer3_b, n_in=hidden_n_in,
                                      n_out=mlp_layers[0], activation=T.tanh, rng=0)

    # layer 4: logit (logistic regression) or SVM
    c_result = []
    c_prob = []
    if classifier == CNN.enums.ClassifierType.logit:
        c_result, c_prob = CNN.logit.classify_images(input_flatten=layer2_output_flattened,
                                                     hidden_output=hiddenLayer.output,
                                                     filters=layer2_filters, W=layer4_W,
                                                     b=layer4_b)
    elif classifier == CNN.enums.ClassifierType.svm:
        c_result, c_prob = CNN.svm.classify_images(input_flatten=layer2_output_flattened,
                                                   hidden_output=hiddenLayer.output,
                                                   filters=layer2_filters, W=layer4_W,
                                                   b=layer4_b)
    else:
        raise TypeError('Unknown classifier type, should be either logit or svm', ('classifier:', classifier))

    end_time = time.clock()

    # that's because we only classified one image
    c_result = c_result[0]
    c_prob = c_prob[0]
    c_duration = end_time - start_time

    # __plot_filters_1(img4D, 1)
    # __plot_filters_1(layer0_filters, 2)
    # __plot_filters_1(layer1_filters, 3)
    # __plot_filters_2(loaded_objects[6], 4)
    # __plot_filters_2(loaded_objects[8], 5)

    print('Classification result: %d in %f sec.' % (c_result, c_duration))
    print('Classification confidence: %s' % (CNN.utils.numpy_to_string(c_prob)))

    return c_result, c_prob, c_duration


def __plot_filters_1(filters, figure_num):
    import matplotlib.pyplot as plt

    # plot original image and first and second components of output
    plt.figure(figure_num)
    plt.gray()
    plt.ion()
    length = filters.shape[1]
    for i in range(0, length):
        plt.subplot(1, length, i + 1)
        plt.axis('off')
        plt.imshow(filters[0, i, :, :])
    plt.show()


def __plot_filters_2(filters, figure_num):
    import matplotlib.pyplot as plt

    # plot original image and first and second components of output
    plt.figure(figure_num)
    plt.gray()
    plt.ion()
    length = filters.shape[0]
    for i in range(0, length):
        plt.subplot(1, length, i + 1)
        plt.axis('off')
        plt.imshow(filters[i, 0, :, :])
    plt.show()


# endregion

# region Train Superclass Classifier

def train_superclass_classifier_28(dataset_path, model_path='', kernel_dim=(5, 5), mlp_layers=(400, 100, 4), nkerns=(40, 100),
                                   pool_size=(2, 2), learning_rates=(0.05, 0.008), momentums=(0.9, 0.95), n_epochs=100):
    # train classifier model using lasagne and nolearn
    # this will classify the traffic signs to their super-class only

    # load the data
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate all subsets in one set as the nolearn will use them to train and validate
    # train_x = numpy.concatenate((numpy.concatenate((dataset[0][0], dataset[1][0])), dataset[2][0]))
    # train_y = numpy.concatenate((numpy.concatenate((dataset[0][1], dataset[1][1])), dataset[2][1]))
    train_x = numpy.concatenate((dataset[0][0], dataset[1][0]))
    train_y = numpy.concatenate((dataset[0][1], dataset[1][1]))

    img_dim = 28
    n_batches = 10
    eval_split = 0.1
    batch_size = int(train_y.shape[0] / n_batches)

    # reshape images so it can be ready for convolution
    train_x = train_x.reshape((train_x.shape[0], 1, img_dim, img_dim))

    #########################################
    #       Build the regression model      #
    #########################################

    print("... building the super-class classification model")
    nn_recognition = nolearn.lasagne.NeuralNet(
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
        input_shape=(None, 1, img_dim, img_dim),
        conv1_num_filters=nkerns[0], conv1_filter_size=(kernel_dim[0], kernel_dim[0]), pool1_pool_size=pool_size,
        dropout1_p=0.1,
        conv2_num_filters=nkerns[1], conv2_filter_size=(kernel_dim[1], kernel_dim[1]), pool2_pool_size=pool_size,
        dropout2_p=0.2,
        hidden3_num_units=mlp_layers[0],
        dropout3_p=0.5,
        hidden4_num_units=mlp_layers[1],
        output_num_units=mlp_layers[2],
        output_nonlinearity=lasagne.nonlinearities.softmax,
        objective_loss_function=lasagne.objectives.categorical_crossentropy,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rates[0])),
        update_momentum=theano.shared(CNN.utils.float32(momentums[0])),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        train_split=nolearn.lasagne.TrainSplit(eval_size=eval_split),
        max_epochs=n_epochs,
        regression=False,
        verbose=3,
        on_epoch_finished=[
            CNN.utils.AdjustVariable('update_learning_rate', start=learning_rates[0], stop=learning_rates[1]),
            CNN.utils.AdjustVariable('update_momentum', start=momentums[0], stop=momentums[1]),
            CNN.utils.EarlyStopping(patience=200),
        ]
    )

    ##############################
    # Train The Regression Model #
    ##############################

    n_minibatches = int(train_y.shape[0] / batch_size)
    # Finally, launch the training loop.
    print("... training the regression model")
    print("... in total, %d samples in training" % (train_y.shape[0]))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))

    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    start_time = time.clock()
    nn_recognition.fit(train_x, train_y)
    end_time = time.clock()

    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    # save the model (optional)
    if len(model_path) > 0:
        with open(model_path, "wb") as f:
            pickle.dump(nn_recognition, f, -1)

    # calculate the error
    # predict = nn_classifier.predict(train_x)
    # error = numpy.sum(numpy.not_equal(predict, train_y))
    # print("... train error: %f" % (error / len(train_y)))

    dummy_variable = 10


def train_superclass_classifier_28_light(dataset_path, model_path='', kernel_dim=(5, 5), mlp_layers=(400, 4), nkerns=(40, 100),
                                         pool_size=(2, 2), learning_rates=(0.05, 0.008), momentums=(0.9, 0.95), n_epochs=100):
    # train classifier model using lasagne and nolearn
    # this will classify the traffic signs to their super-class only

    # load the data
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate all subsets in one set as the nolearn will use them to train and validate
    # train_x = numpy.concatenate((numpy.concatenate((dataset[0][0], dataset[1][0])), dataset[2][0]))
    # train_y = numpy.concatenate((numpy.concatenate((dataset[0][1], dataset[1][1])), dataset[2][1]))
    train_x = numpy.concatenate((dataset[0][0], dataset[1][0]))
    train_y = numpy.concatenate((dataset[0][1], dataset[1][1]))
    del dataset

    img_dim = 28
    n_batches = 10
    eval_split = 0.1
    batch_size = int(train_y.shape[0] / n_batches)

    # reshape images so it can be ready for convolution
    train_x = train_x.reshape((train_x.shape[0], 1, img_dim, img_dim))

    #########################################
    #       Build the regression model      #
    #########################################

    print("... building the super-class classification model")
    nn_recognition = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('conv1', lasagne.layers.Conv2DLayer),
            ('pool1', lasagne.layers.MaxPool2DLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('conv2', lasagne.layers.Conv2DLayer),
            ('pool2', lasagne.layers.MaxPool2DLayer),
            ('dropout2', lasagne.layers.DropoutLayer),
            ('hidden3', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=(None, 1, img_dim, img_dim),
        conv1_num_filters=nkerns[0], conv1_filter_size=(kernel_dim[0], kernel_dim[0]), pool1_pool_size=pool_size,
        dropout1_p=0.1,
        conv2_num_filters=nkerns[1], conv2_filter_size=(kernel_dim[1], kernel_dim[1]), pool2_pool_size=pool_size,
        dropout2_p=0.2,
        hidden3_num_units=mlp_layers[0],
        output_num_units=mlp_layers[1],
        output_nonlinearity=lasagne.nonlinearities.softmax,
        objective_loss_function=lasagne.objectives.categorical_crossentropy,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rates[0])),
        update_momentum=theano.shared(CNN.utils.float32(momentums[0])),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        train_split=nolearn.lasagne.TrainSplit(eval_size=eval_split),
        max_epochs=n_epochs,
        regression=False,
        verbose=3,
        on_epoch_finished=[
            CNN.utils.AdjustVariable('update_learning_rate', start=learning_rates[0], stop=learning_rates[1]),
            CNN.utils.AdjustVariable('update_momentum', start=momentums[0], stop=momentums[1]),
            CNN.utils.EarlyStopping(patience=200),
        ]
    )

    ##############################
    # Train The Regression Model #
    ##############################

    n_minibatches = int(train_y.shape[0] / batch_size)
    # Finally, launch the training loop.
    print("... training the regression model")
    print("... in total, %d samples in training" % (train_y.shape[0]))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))

    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    start_time = time.clock()
    nn_recognition.fit(train_x, train_y)
    end_time = time.clock()

    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    # save the model (optional)
    if len(model_path) > 0:
        with open(model_path, "wb") as f:
            pickle.dump(nn_recognition, f, -1)

    dummy_variable = 10


def train_superclass_classifier_32(dataset_path, model_path='', kernel_dim=(5, 7, 4), mlp_layers=(7200 / 4, 7200 / 8, 4), nkerns=(10, 50, 200),
                                   pool_size=(2, 2), learning_rates=(0.05, 0.008), momentums=(0.9, 0.95), n_epochs=100):
    # train classifier model using lasagne and nolearn
    # this will classify the traffic signs to their super-class only

    # load the data
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate all subsets in one set as the nolearn will use them to train and validate
    train_x = numpy.concatenate((numpy.concatenate((dataset[0][0], dataset[1][0])), dataset[2][0]))
    train_y = numpy.concatenate((numpy.concatenate((dataset[0][1], dataset[1][1])), dataset[2][1]))

    img_dim = 80
    n_batches = 10
    eval_split = 0.1
    batch_size = int(train_y.shape[0] / n_batches)

    # reshape images so it can be ready for convolution
    train_x = train_x.reshape((train_x.shape[0], 1, img_dim, img_dim))

    #########################################
    #       Build the regression model      #
    #########################################

    print("... building the super-class classification model")
    nn_recognition = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('conv1', lasagne.layers.Conv2DLayer),
            ('pool1', lasagne.layers.MaxPool2DLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('conv2', lasagne.layers.Conv2DLayer),
            ('pool2', lasagne.layers.MaxPool2DLayer),
            ('dropout2', lasagne.layers.DropoutLayer),
            ('conv3', lasagne.layers.Conv2DLayer),
            ('pool3', lasagne.layers.MaxPool2DLayer),
            ('dropout3', lasagne.layers.DropoutLayer),
            ('hidden4', lasagne.layers.DenseLayer),
            ('dropout4', lasagne.layers.DropoutLayer),
            ('hidden5', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=(None, 1, img_dim, img_dim),
        conv1_num_filters=nkerns[0], conv1_filter_size=(kernel_dim[0], kernel_dim[0]), pool1_pool_size=pool_size,
        dropout1_p=0.1,
        conv2_num_filters=nkerns[1], conv2_filter_size=(kernel_dim[1], kernel_dim[1]), pool2_pool_size=pool_size,
        dropout2_p=0.2,
        conv3_num_filters=nkerns[2], conv3_filter_size=(kernel_dim[2], kernel_dim[2]), pool3_pool_size=pool_size,
        dropout3_p=0.3,
        hidden4_num_units=mlp_layers[0],
        dropout4_p=0.5,
        hidden5_num_units=mlp_layers[1],
        output_num_units=mlp_layers[2],
        output_nonlinearity=lasagne.nonlinearities.softmax,
        objective_loss_function=lasagne.objectives.categorical_crossentropy,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rates[0])),
        update_momentum=theano.shared(CNN.utils.float32(momentums[0])),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        train_split=nolearn.lasagne.TrainSplit(eval_size=eval_split),
        max_epochs=n_epochs,
        regression=False,
        verbose=3,
        on_epoch_finished=[
            CNN.utils.AdjustVariable('update_learning_rate', start=learning_rates[0], stop=learning_rates[1]),
            CNN.utils.AdjustVariable('update_momentum', start=momentums[0], stop=momentums[1]),
            CNN.utils.EarlyStopping(patience=200),
        ]
    )

    ##############################
    # Train The Regression Model #
    ##############################

    n_minibatches = int(train_y.shape[0] / batch_size)
    # Finally, launch the training loop.
    print("... training the regression model")
    print("... in total, %d samples in training" % (train_y.shape[0]))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))

    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    start_time = time.clock()
    nn_recognition.fit(train_x, train_y)
    end_time = time.clock()

    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    # save the model (optional)
    if len(model_path) > 0:
        with open(model_path, "wb") as f:
            pickle.dump(nn_recognition, f, -1)

    # calculate the error
    # predict = nn_classifier.predict(train_x)
    # error = numpy.sum(numpy.not_equal(predict, train_y))
    # print("... train error: %f" % (error / len(train_y)))

    dummy_variable = 10


def train_superclass_classifier_80(dataset_path, model_path='', kernel_dim=(9, 7, 4), mlp_layers=(7200 / 4, 7200 / 8, 3), nkerns=(10, 50, 200),
                                   pool_size=(2, 2), learning_rates=(0.05, 0.008), momentums=(0.9, 0.95), n_epochs=100):
    # train classifier model using lasagne and nolearn
    # this will classify the traffic signs to their super-class only

    # load the data
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate all subsets in one set as the nolearn will use them to train and validate
    train_x = numpy.concatenate((numpy.concatenate((dataset[0][0], dataset[1][0])), dataset[2][0]))
    train_y = numpy.concatenate((numpy.concatenate((dataset[0][1], dataset[1][1])), dataset[2][1]))

    img_dim = 80
    n_batches = 10
    eval_split = 0.1
    batch_size = int(train_y.shape[0] / n_batches)

    # reshape images so it can be ready for convolution
    train_x = train_x.reshape((train_x.shape[0], 1, img_dim, img_dim))

    #########################################
    #       Build the regression model      #
    #########################################

    print("... building the super-class classification model")
    nn_recognition = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('conv1', lasagne.layers.Conv2DLayer),
            ('pool1', lasagne.layers.MaxPool2DLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('conv2', lasagne.layers.Conv2DLayer),
            ('pool2', lasagne.layers.MaxPool2DLayer),
            ('dropout2', lasagne.layers.DropoutLayer),
            ('conv3', lasagne.layers.Conv2DLayer),
            ('pool3', lasagne.layers.MaxPool2DLayer),
            ('dropout3', lasagne.layers.DropoutLayer),
            ('hidden4', lasagne.layers.DenseLayer),
            ('dropout4', lasagne.layers.DropoutLayer),
            ('hidden5', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=(None, 1, img_dim, img_dim),
        conv1_num_filters=nkerns[0], conv1_filter_size=(kernel_dim[0], kernel_dim[0]), pool1_pool_size=pool_size,
        dropout1_p=0.1,
        conv2_num_filters=nkerns[1], conv2_filter_size=(kernel_dim[1], kernel_dim[1]), pool2_pool_size=pool_size,
        dropout2_p=0.2,
        conv3_num_filters=nkerns[2], conv3_filter_size=(kernel_dim[2], kernel_dim[2]), pool3_pool_size=pool_size,
        dropout3_p=0.3,
        hidden4_num_units=mlp_layers[0],
        dropout4_p=0.5,
        hidden5_num_units=mlp_layers[1],
        output_num_units=mlp_layers[2],
        output_nonlinearity=lasagne.nonlinearities.softmax,
        objective_loss_function=lasagne.objectives.categorical_crossentropy,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rates[0])),
        update_momentum=theano.shared(CNN.utils.float32(momentums[0])),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        train_split=nolearn.lasagne.TrainSplit(eval_size=eval_split),
        max_epochs=n_epochs,
        regression=False,
        verbose=3,
        on_epoch_finished=[
            CNN.utils.AdjustVariable('update_learning_rate', start=learning_rates[0], stop=learning_rates[1]),
            CNN.utils.AdjustVariable('update_momentum', start=momentums[0], stop=momentums[1]),
            CNN.utils.EarlyStopping(patience=200),
        ]
    )

    ##############################
    # Train The Regression Model #
    ##############################

    n_minibatches = int(train_y.shape[0] / batch_size)
    # Finally, launch the training loop.
    print("... training the regression model")
    print("... in total, %d samples in training" % (train_y.shape[0]))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))

    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    start_time = time.clock()
    nn_recognition.fit(train_x, train_y)
    end_time = time.clock()

    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    # save the model (optional)
    if len(model_path) > 0:
        with open(model_path, "wb") as f:
            pickle.dump(nn_recognition, f, -1)

    # calculate the error
    # predict = nn_classifier.predict(train_x)
    # error = numpy.sum(numpy.not_equal(predict, train_y))
    # print("... train error: %f" % (error / len(train_y)))

    dummy_variable = 10


# endregion

# region Test Superclass Classifier

def classify_superclass_from_database(model_path, dataset_path, img_dim):
    data = pickle.load(open(dataset_path, "rb"))

    # load the regression model
    with open(model_path, 'rb') as f:
        net_cnn = pickle.load(f)

    for i in range(0, 3):
        images = data[i][0]
        classes = data[i][1]

        # read the image, don't forget to reshape
        # the image for the sake of the detector
        imgs = numpy.asarray(images)
        imgs = imgs.reshape((imgs.shape[0], 1, img_dim, img_dim))
        prediction = net_cnn.predict(imgs)

        error = 100 * numpy.sum(numpy.not_equal(prediction, classes)) / len(classes)
        print("... error percentage: %f%%" % (error))


def classify_superclass_from_database_1(model_path, dataset_path, img_dim):
    data = pickle.load(open(dataset_path, "rb"))

    # load the regression model
    with open(model_path, 'rb') as f:
        net_cnn = pickle.load(f)

    images = data[0]
    classes = data[1]

    # read the image, don't forget to reshape
    # the image for the sake of the detector
    imgs = numpy.asarray(images)
    imgs = imgs.reshape((imgs.shape[0], 1, img_dim, img_dim))
    prediction = net_cnn.predict(imgs)

    error = 100 * numpy.sum(numpy.not_equal(prediction, classes)) / len(classes)
    print("... error percentage: %f%%" % (error))


def classify_superclass_from_image(model_path, img_dim):
    import io
    import os
    import os.path

    superclass_id = 1
    ##directory = "D:\\_Dataset\\SuperClass\\Test_Preprocessed_Revised_28\\0000%d\\" % (superclass_id)
    directory = "D:\\_Dataset\\BelgiumTS\\Training_Preprocessed_28\\00032\\"

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    imgs = []
    for file in files:
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        imgs.append(img)

    # read the image, don't forget to reshape
    # the image for the sake of the detector
    imgs = numpy.asarray(imgs)
    imgs = imgs.reshape((imgs.shape[0], 1, img_dim, img_dim))

    # load the regression model
    with open(model_path, 'rb') as f:
        net_cnn = pickle.load(f)

    prediction = net_cnn.predict(imgs)

    error = 100 * len(prediction[prediction != superclass_id]) / len(imgs)
    print("... error percentage: %f%%" % error)
    dummy = 10

# endregion
