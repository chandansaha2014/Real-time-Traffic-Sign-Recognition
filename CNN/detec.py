import os
import sys
import time

import cv2
import skimage
import skimage.transform
import skimage.exposure

import numpy
import theano
import theano.tensor as T

import nolearn
import nolearn.lasagne
import lasagne

import pickle
import CNN
import CNN.svm
import CNN.logit
import CNN.utils
import CNN.mlp
import CNN.conv
import CNN.enums
import CNN.recog
import CNN.nms
import CNN.prop

from CNN.mlp import HiddenLayer

# region Train Detector

def train_model_28(dataset_path, recognition_model_path, detection_model_path='', learning_rate=0.1, n_epochs=10, batch_size=500,
                  classifier=CNN.enums.ClassifierType.logit):
    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    n_valid_batches = int(n_valid_batches / batch_size)
    n_test_batches = int(n_test_batches / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.imatrix('y')

    # load model and read it's parameters
    # the same weights of the convolutional layers will be used
    # in training the detector
    loaded_objects = CNN.utils.load_model(recognition_model_path)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]
    rng = numpy.random.RandomState(23455)

    # the number of layers in the MLP classifier to train is not optional
    # the input of the first MLP layer has to be compatible with the output of conv layers
    # while the output of the last MLP layer has to comprise the img_dim because at the end
    # of the day the MLP results is classes each of them represent a pixel
    # this is the regression fashion of the MLP, each class represents the pixel, i.e what
    # is the pixel of the predicted region
    mlp_layers = loaded_objects[4]
    mlp_layers = (mlp_layers[0], (img_dim + 1))

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)

    # first, filter the given input images using the weights of the filters
    # of the given class_model_path
    # then, train a mlp as a regression model not classification
    # then save all of the cnn_model and the regression_model into a file 'det_model_path'

    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)

    # layer 0: Conv-Pool
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))
    layer0 = CNN.conv.ConvPoolLayerTrained(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # layer 1: Conv-Pool
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)  # = 12 in case of mnist
    layer1_kernel_dim = kernel_dim[1]
    layer1 = CNN.conv.ConvPoolLayerTrained(
        input=layer0.output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # Layer 2: the HiddenLayer being fully-connected, it operates on 2D matrices
    layer2 = CNN.mlp.HiddenLayer(
        rng,
        input=layer1.output.flatten(2),
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # Layer 3: classify the values of the fully-connected sigmoidal layer
    layer3_n_outs = [1] * 4
    layer3 = CNN.logit.MultiLinearRegression(input=layer2.output, n_in=mlp_layers[0], n_outs=layer3_n_outs)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.cost(y)

    # experimental, add L1, L2 regularization to the regressor
    # self.L1 = (
    #         abs(self.hiddenLayer.W).sum()
    #         + abs(self.logRegressionLayer.W).sum()
    #     )
    #
    #     # square of L2 norm ; one regularization option is to enforce
    #     # square of L2 norm to be small
    #     self.L2_sqr = (
    #         (self.hiddenLayer.W ** 2).sum()
    #         + (self.logRegressionLayer.W ** 2).sum()
    #     )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params

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
            minibatch_avg_cost = train_model(minibatch_index)

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

    if len(detection_model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(detection_model_path, 'wb')
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


def train_model_80(dataset_path, recognition_model_path, detection_model_path='', learning_rate=0.1, momentum=0.9,
               n_epochs=10, batch_size=500, mlp_layers=(1000, 4)):
    # do all the cov+pool computation using theano
    # while train the regressor of the detector using nolearn and lasagne
    # don't forget to operate on batches becuase:
    # 1. you can't convolve all the training image in once shot
    # 2. to train better regressor

    # load model and read it's parameters
    # the same weights of the convolutional layers will be used
    # in training the detector
    loaded_objects = CNN.utils.load_model(recognition_model_path, CNN.enums.ModelType._02_conv3_mlp2)
    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]

    # load the data and normalize the target to be from range [-1, 1]
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate validation and training sets
    train_x = numpy.concatenate((dataset[0][0], dataset[1][0]))
    train_y = numpy.concatenate((dataset[0][1], dataset[1][1])).astype("float32")
    train_y = ((train_y * 2) - img_dim) / img_dim

    # first, filter the given input images using the weights of the filters
    # of the given class_model_path
    # then, train a mlp as a regression model not classification
    # then save all of the cnn_model and the regression_model into a file 'det_model_path'
    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    layer0_input = T.tensor4('input')
    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)
    layer3_input_shape = (batch_size, nkerns[2] * layer3_img_dim * layer3_img_dim)

    # layer 0, 1, 2: Conv-Pool
    layer0_output = CNN.conv.convpool_layer(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        pool_size=pool_size
    )
    layer1_output = CNN.conv.convpool_layer(
        input=layer0_output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        pool_size=pool_size
    )
    layer2_output = CNN.conv.convpool_layer(
        input=layer1_output, W=layer2_W, b=layer2_b,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        pool_size=pool_size
    )
    # do the filtering using 3 layers of Conv+Pool
    conv_fn = theano.function([layer0_input], layer2_output)

    #########################################
    #       Build the regression model      #
    #########################################
    print("... building the regression model")
    nn_regression = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('hidden1', lasagne.layers.DenseLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('hidden2', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=layer3_input_shape,
        hidden1_num_units=mlp_layers[0],
        dropout1_p=0.5,
        hidden2_num_units=int(mlp_layers[0] / 2),
        output_num_units=mlp_layers[1], output_nonlinearity=None,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rate)),
        update_momentum=theano.shared(CNN.utils.float32(momentum)),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        train_split=nolearn.lasagne.TrainSplit(eval_size=0.0),
        regression=True,
        max_epochs=1,
        verbose=1,
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
    start_time = time.clock()
    # We iterate over epochs:
    for epoch in range(n_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for batch in CNN.utils.iterate_minibatches(train_x, train_y, batch_size):
            train_batches += 1
            inputs, targets = batch
            inputs_reshaped = inputs.reshape(-1, 1, layer0_img_dim, layer0_img_dim)
            filters = conv_fn(inputs_reshaped)
            filters = filters.reshape(layer3_input_shape).astype("float32")
            nn_regression.fit(filters, targets)
            print("... epoch: %d/%d, mini-batch: %d/%d" % (epoch + 1, n_epochs, train_batches, n_minibatches))

        # for more tuning, decrease learning rate and increase momentum
        # after every epoch
        # nn_regression.set_params()
        momentum *= 1.05
        learning_rate *= 0.95

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f" % (duration))
    with open(detection_model_path, "wb") as f:
        pickle.dump(nn_regression, f, -1)


def train_regressor(dataset_path, detection_model_path='', learning_rate=0.02, momentum=0.9,
                    n_epochs=50, mlp_layers=(7200, 4)):
    # train only regression model using lasagne and nolearn
    # we will depend on the already convolutioned images
    # i.e the filters as input to the regression model

    img_dim = 80

    # load the data and normalize the target to be from range [-1, 1]
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate all subsets in one set as the nolearn will use them
    # to train and validate
    train_x = numpy.concatenate((numpy.concatenate((dataset[0][0], dataset[1][0])), dataset[2][0]))
    train_y = numpy.concatenate((numpy.concatenate((dataset[0][1], dataset[1][1])), dataset[2][1])).astype("float32")
    train_y = ((train_y * 2) - img_dim) / img_dim

    n_batches = 10
    batch_size = int(train_y.shape[0] / n_batches)
    layer3_input_shape = (batch_size, train_x.shape[1])

    #########################################
    #       Build the regression model      #
    #########################################
    print("... building the regression model")
    nn_regression = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('hidden1', lasagne.layers.DenseLayer),
            ('hidden2', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=layer3_input_shape,
        hidden1_num_units=int(mlp_layers[0] / 4),
        hidden2_num_units=int(mlp_layers[0] / 8),
        output_num_units=mlp_layers[1], output_nonlinearity=None,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rate)),
        update_momentum=theano.shared(CNN.utils.float32(momentum)),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        train_split=nolearn.lasagne.TrainSplit(eval_size=0.1),
        regression=True,
        max_epochs=n_epochs,
        verbose=1,
        on_epoch_finished=[
            CNN.utils.AdjustVariable('update_learning_rate', start=0.05, stop=0.008),
            CNN.utils.AdjustVariable('update_momentum', start=momentum, stop=0.95),
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
    start_time = time.clock()
    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    nn_regression.fit(train_x, train_y)

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))
    with open(detection_model_path, "wb") as f:
        pickle.dump(nn_regression, f, -1)


def train_binary_detector(dataset_path, detection_model_path='', learning_rate=0.02, momentum=0.9,
                          n_epochs=50, mlp_layers=(1800, 900, 1)):
    # train only binary classifier model using lasagne and nolearn
    # we will depend on the already convolutioned images
    # i.e the filters as input to the regression model
    # the sole purpose of the model is to tell if the image belongs
    # to the superclass or not

    img_dim = 80

    # load the data and normalize the target to be from range [-1, 1]
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate all subsets in one set as the nolearn will use them
    # to train and validate
    train_x = numpy.concatenate((numpy.concatenate((dataset[0][0], dataset[1][0])), dataset[2][0]))
    train_y = numpy.concatenate((numpy.concatenate((dataset[0][1], dataset[1][1])), dataset[2][1]))

    # convert target to binary
    train_y = train_y.astype(bool)

    eval_size = 0.1
    n_batches = 10
    batch_size = int(train_y.shape[0] / n_batches)
    layer0_input_shape = (None, train_x.shape[1])

    #########################################
    #       Build the regression model      #
    #########################################
    print("... building the binary detection model")
    nn_regression = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('hidden1', lasagne.layers.DenseLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('hidden2', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=layer0_input_shape,
        hidden1_num_units=mlp_layers[0],
        dropout1_p=0.5,
        hidden2_num_units=mlp_layers[1],
        output_num_units=mlp_layers[2],
        output_nonlinearity=lasagne.nonlinearities.sigmoid,
        objective_loss_function=lasagne.objectives.binary_crossentropy,
        update_learning_rate=theano.shared(CNN.utils.float32(learning_rate)),
        update_momentum=theano.shared(CNN.utils.float32(momentum)),
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=batch_size),
        train_split=nolearn.lasagne.TrainSplit(eval_size=eval_size),
        max_epochs=n_epochs,
        regression=True,
        verbose=3,
        on_epoch_finished=[
            CNN.utils.AdjustVariable('update_learning_rate', start=0.05, stop=0.008),
            CNN.utils.AdjustVariable('update_momentum', start=momentum, stop=0.95),
            CNN.utils.EarlyStopping(patience=200),
        ]
    )

    ##############################
    # Train The Regression Model #
    ##############################
    n_minibatches = int(train_y.shape[0] / batch_size)
    # Finally, launch the training loop.
    print("... training the binary detection model")
    print("... in total, %d samples in training" % (train_y.shape[0]))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))

    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    start_time = time.clock()
    nn_regression.fit(train_x, train_y)
    end_time = time.clock()

    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    # save the model
    with open(detection_model_path, "wb") as f:
        pickle.dump(nn_regression, f, -1)

    # calculate the error
    predict = nn_regression.predict(train_x)

    # convert predication from float to binary
    # as we've trained the network using regression flag = true
    predict = predict.reshape((predict.shape[0],))
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0
    predict = predict.astype(bool)

    error = numpy.sum(numpy.not_equal(predict, train_y))
    print("... train error: %f" % (error / len(train_y)))


# endregion

# region Train Detector From Scratch

def train_from_scatch_regressor(dataset_path, detection_model_path, learning_rate=0.1, n_epochs=10, batch_size=500,
                                nkerns=(40, 40 * 9), mlp_layers=(800, 29), kernel_dim=(5, 5), img_dim=28, pool_size=(2, 2)):
    datasets = CNN.utils.load_data(dataset_path)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    n_valid_batches = int(n_valid_batches / batch_size)
    n_test_batches = int(n_test_batches / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.imatrix('y')
    rng = numpy.random.RandomState(23455)

    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer0_input = x.reshape((batch_size, 1, layer0_img_dim, layer0_img_dim))
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer3_n_outs = [mlp_layers[1]] * 4

    # layer 0: Conv-Pool
    layer0 = CNN.conv.ConvPoolLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        poolsize=pool_size
    )

    # layer 1: Conv-Pool
    layer1 = CNN.conv.ConvPoolLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        poolsize=pool_size
    )

    # Layer 2: the HiddenLayer being fully-connected, it operates on 2D matrices
    layer2 = HiddenLayer(
        rng,
        input=layer1.output.flatten(2),
        n_in=nkerns[1] * layer2_img_dim * layer2_img_dim,
        n_out=mlp_layers[0],
        activation=T.tanh
    )

    # Layer 3: classify the values of the fully-connected sigmoidal layer
    layer3 = CNN.logit.MultiLogisticRegression(input=layer2.output, n_in=mlp_layers[0], n_outs=layer3_n_outs)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # experimental, add L1, L2 regularization to the regressor
    # self.L1 = (
    #         abs(self.hiddenLayer.W).sum()
    #         + abs(self.logRegressionLayer.W).sum()
    #     )
    #
    #     # square of L2 norm ; one regularization option is to enforce
    #     # square of L2 norm to be small
    #     self.L2_sqr = (
    #         (self.hiddenLayer.W ** 2).sum()
    #         + (self.logRegressionLayer.W ** 2).sum()
    #     )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y, mlp_layers[1]),
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
            minibatch_avg_cost = train_model(minibatch_index)

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

    if len(detection_model_path) == 0:
        return

    # serialize the params of the model
    # the -1 is for HIGHEST_PROTOCOL
    # this will overwrite current contents and it triggers much more efficient storage than numpy's default
    save_file = open(detection_model_path, 'wb')
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


def train_from_scatch_binary_detector(dataset_path, model_path='', kernel_dim=(9, 7, 4), mlp_layers=(1800, 900, 2), nkerns=(10, 50, 200),
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
    nn_detection = nolearn.lasagne.NeuralNet(
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
    print("... training the detection model")
    print("... in total, %d samples in training" % (train_y.shape[0]))
    print("... since we have batch size of %d" % (batch_size))
    print("... then training will run for %d mini-batches and for %d epochs" % (n_minibatches, n_epochs))

    # no need to iterate on epochs or mini-baches because fitting a nolearn network
    # takes care of all of that if configured correctly
    start_time = time.clock()
    nn_detection.fit(train_x, train_y)
    end_time = time.clock()

    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min" % (duration))

    # save the model (optional)
    if len(model_path) > 0:
        with open(model_path, "wb") as f:
            pickle.dump(nn_detection, f, -1)

    # calculate the error
    predict = nn_detection.predict(train_x)
    error = numpy.sum(numpy.not_equal(predict, train_y))
    print("... train error: %f" % (error / len(train_y)))

    dummy_variable = 10


# endregion

# region Do Recognition

def detect_from_dataset(dataset_path, recognition_model_path, detection_model_path):
    # load the cnn model to run the extract convolution filters from the images
    # then load the regression model to run on these filters
    # note, you may apply on train, valid and test datasets for comparison

    # load model and read it's parameters
    loaded_objects = CNN.utils.load_model(recognition_model_path, CNN.enums.ModelType._02_conv3_mlp2)
    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]
    batch_size = 1000

    # parameters of the convolutional+maxpool layers
    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    layer0_input = T.tensor4('input')
    layer0_img_dim = img_dim
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)
    layer3_input_shape = (batch_size, nkerns[2] * layer3_img_dim * layer3_img_dim)

    # layer 0, 1, 2: Conv-Pool
    layer0_output = CNN.conv.convpool_layer(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        pool_size=pool_size
    )
    layer1_output = CNN.conv.convpool_layer(
        input=layer0_output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        pool_size=pool_size
    )
    layer2_output = CNN.conv.convpool_layer(
        input=layer1_output, W=layer2_W, b=layer2_b,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        pool_size=pool_size
    )
    # do the filtering using 3 layers of Conv+Pool
    conv_fn = theano.function([layer0_input], layer2_output)

    #########################################
    #       Build the regression model      #
    #########################################
    print("... load the regression model")
    with open(detection_model_path, 'rb') as f:
        nn_regression = pickle.load(f)

    ##############################
    # Test the regression model  #
    ##############################
    # load the data and normalize the target to be from range [-1, 1]
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print('... start predicting')
    start_time = time.clock()
    sub_count = 0
    for subset in dataset:
        sub_count += 1
        subset_x, subset_y_int = subset
        subset_x = subset_x[0:batch_size]
        subset_y_int = subset_y_int[0:batch_size]
        subset_y = ((subset_y_int * 2) - img_dim) / img_dim
        inputs_reshaped = subset_x.reshape(batch_size, 1, layer0_img_dim, layer0_img_dim)
        filters = conv_fn(inputs_reshaped)
        filters = filters.reshape(layer3_input_shape).astype("float32")
        predict_y = nn_regression.predict(filters)
        # calculate the error of the prediction
        predict_y = numpy.rint(((predict_y * img_dim) + img_dim) / 2).astype(int)
        error = numpy.mean(numpy.mean(numpy.abs(subset_y_int - predict_y), axis=0))
        print("... error for subset %d is: %f pixels" % (sub_count, error))

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    print("... finish training the model, total time consumed: %f min." % (duration))


def binary_detect_from_file_fast(img_path, recognition_model_path, detection_model_path, img_dim):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    ##############################
    # Extract and detect regions #
    ##############################

    # this is similar to "binary_detect_from_file" except we won't slide the window
    # we'll depend on the detection proposals then take ~5 regions with the same center
    # as the proposal but at different scales
    # this results in much much faster process time
    # pre-process image by: equalize histogram and stretch intensity
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img = img.astype(float) / 255.0

    # min, max defines what is the range of scaling the sliding window
    # while scaling factor controls the amount of pixels to go down in each scale
    max_window_dim = int(img_dim * 2)
    min_window_dim = int(img_dim / 4)

    img_shape = img.shape
    img_width = img_shape[1]
    img_height = img_shape[0]

    r_count = 0

    # regions, locations and window_dim at each scale
    regions = []
    locations = []
    window_dims = []

    # important, instead of naively add every sliding window, we'll only add
    # windows that covers the strong detection proposals
    prop_weak, prop_strong, prop_map, prop_circles = CNN.prop.detection_proposal(img_color, min_dim=min_window_dim, max_dim=max_window_dim)
    if len(prop_strong) == 0:
        print("... NO TRAFFIC SIGN PROPOSALS WERE FOUND")
        return

    # loop on the detection proposals
    scales = numpy.arange(0.7, 1.58, 0.05)
    for prop in prop_strong:
        x1 = prop[0]
        y1 = prop[1]
        x2 = prop[2]
        y2 = prop[3]
        w = x2 - x1
        h = y2 - y1
        window_dim = max(h, w)
        center_x = int(x1 + round(w / 2))
        center_y = int(y1 + round(h / 2))

        for scale in scales:
            dim = window_dim * scale
            dim_half = round(dim / 2)
            dim = round(dim)
            x1 = center_x - dim_half
            y1 = center_y - dim_half
            x2 = center_x + dim_half
            y2 = center_y + dim_half

            # pre-process the region and scale it to img_dim
            region = img[y1:y2, x1:x2]
            region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))
            region = skimage.exposure.equalize_hist(region)

            # we only need to store the region, it's top-left corner and sliding window dim
            regions.append(region)
            locations.append([x1, y1])
            window_dims.append(dim)

    regions = numpy.asarray(regions)
    locations = numpy.asarray(locations)
    window_dims = numpy.asarray(window_dims)

    # run the detector on the regions
    start_time = time.clock()
    predictions = __detect_from_scales_regions(recognition_model_path, detection_model_path, regions, False)

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    strong_regions = []

    print("... detection regions: %d, duration(min.): %f" % (r_count, duration))

    # construct the probability map for each scale and show it/ save it
    s_count = 0
    overlap_thresh = 0.5
    min_overlap = 0
    for pred, loc, window_dim in zip(predictions, locations, window_dims):
        s_count += 1
        map, w_regions, s_regions = __probability_map(img, [pred], [loc], window_dim, img_width, img_height, img_dim, s_count,
                                                      False, overlap_thresh=overlap_thresh, min_overlap=min_overlap)
        if len(s_regions) > 0:
            strong_regions.append(s_regions)
            print("Scale: %d, window_dim: %d, regions: %d, strong regions detected" % (s_count, window_dim, r_count))
        else:
            print("Scale: %d, window_dim: %d, regions: %d, no regions detected" % (s_count, window_dim, r_count))

    # now, after we finished scanning at all the levels, we should make the final verdict
    # by suppressing all the strong_regions that we extracted on different scales
    if len(strong_regions) > 0:
        overlap_thresh = 0.35
        min_overlap = round(len(scales) - 9 / 10)
        regions = numpy.vstack(strong_regions)
        __confidence_map(img, img_width, img_height, regions, s_count, overlap_thresh=overlap_thresh, min_overlap=min_overlap)


def binary_detect_from_file(img_path, recognition_model_path, detection_model_path, img_dim, proposals=True):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    ##############################
    # Extract and detect regions #
    ##############################

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also down_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa

    # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
    # that means, we'll start with a window with initial size 320*320
    # for each ground_truth boundary, extract regions in such that:
    # 1. each region fully covers the boundary
    # 2. the boundary must not be smaller than the 1/5 of the region
    # ie. according to initial window size, the boundary must not be smaller than 80*80
    # but sure we will recognize smaller ground truth because we down_scale the window every step
    # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
    # don't forget that stride of sliding the window is dynamic

    # pre-process image by: equalize histogram and stretch intensity
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # img_proc = skimage.exposure.equalize_hist(img)
    # img_proc = skimage.exposure.equalize_adapthist(img_proc, clip_limit=0.05, kernel_size=(8, 8))
    # img_proc = skimage.exposure.rescale_intensity(img_proc, in_range=(0.2, 0.75))
    img = img.astype(float) / 255.0

    # min, max defines what is the range of scaling the sliding window
    # while scaling factor controls the amount of pixels to go down in each scale
    max_window_dim = int(img_dim * 2)
    min_window_dim = int(img_dim / 4)
    down_scale_factor = 0.9
    stride_factor = 0.1

    img_shape = img.shape
    img_width = img_shape[1]
    img_height = img_shape[0]

    s_count = 0
    r_count = 0

    # regions, locations and window_dim at each scale
    scale_regions = []
    scale_locations = []
    scale_window_dim = []

    # we start by the window dimension = max and stop when it goes below
    # the min, at each iteration, we scale down the window dimension by a factor
    window_dim = max_window_dim

    # scale_down until you reach the min window
    # instead of scaling up the image itself, we scale down the sliding window
    while window_dim >= min_window_dim:

        # we need to save window_dim at each scale to resize back the predicted region
        scale_window_dim.append(window_dim)

        # locations are the x,y position (top left) of the sliding-windows
        # regions are the extracted sliding windows from the image, passed
        # later to the detector to predict the location of the traffic sign with-in each one
        regions = []
        locations = []

        # stride is dynamic, smaller strides for smaller scales
        # this means that stride is equivialant to 2 pixels
        # when the window is resized to the img_dim (required for CNN)
        # r_factor = window_dim / min_window_dim
        # stride = int(stride_factor * int(r_factor))

        # simpler way to calculate the stride is: the stride is 10% of the current window dim
        stride = int(window_dim * stride_factor)

        s_count += 1
        r_count = 0

        # for the current scale of the window, extract regions, start from the
        y = 0
        x_count = 0
        y_count = 0
        region_shape = []

        # check if option of using proposal is enabled
        if proposals:
            # important, instead of naively add every sliding window, we'll only add
            # windows that covers the strong detection proposals
            prop_max_dim = int(window_dim * 1.1)
            prop_min_dim = int(window_dim * 0.65)
            prop_weak, prop_strong, prop_map, prop_circles = CNN.prop.detection_proposal(img_color, min_dim=prop_min_dim, max_dim=prop_max_dim)
            if len(prop_strong) == 0:
                print("Scale: %d, stride: %d, window_dim: %d, regions: %d" % (s_count, stride, window_dim, r_count))
                window_dim = int(window_dim * down_scale_factor)
                scale_regions.append([])
                scale_locations.append([])
                continue

        while y <= img_height:
            x = 0
            x_count = 0
            while x <= img_width:

                # check if option of using proposal is enabled
                if proposals:
                    # if the current region does not intersect with the proposal, then ignore it
                    if not numpy.any(prop_map[y:y + window_dim, x:x + window_dim]):
                        x += stride
                        region_shape = (window_dim, window_dim)
                        continue

                # - add region to the region list
                # - adjust the position of the ground_truth to be relative to the window
                #   relative frame of reference (i.e not relative to the image)
                # - don't forget to re_scale the extracted/sampled region to be 28*28
                #   hence, multiply the relative position with this scaling accordingly
                # - also, the image needs to be preprocessed so it can be ready for the CNN
                region = img[y:y + window_dim, x:x + window_dim]
                region_shape = region.shape
                region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))

                # pre-process the region if needed
                region = skimage.exposure.equalize_hist(region)

                # we only need to store the region, it's top-left corner and sliding window dim
                regions.append(region)
                locations.append([x, y])

                r_count += 1

                # # # save region for experiemnt
                # filePathWrite = "D:\\_Dataset\\GTSDB\\Test_Regions\\%s_%s.png" % ("{0:03d}".format(s_count), "{0:03d}".format(r_count))
                # img_save = region * 255
                # img_save = img_save.astype(int)
                # cv2.imwrite(filePathWrite, img_save)

                x_count += 1
                x += stride
                if region_shape[1] < window_dim:
                    break

            y_count += 1
            if region_shape[0] < window_dim:
                break
            y += stride

        # append all the regions extracted from the current scale
        # we'll only do detection after we collect all the regions
        # from all the scales
        scale_regions.append(regions)
        scale_locations.append(locations)

        print("Scale: %d, stride: %d, window_dim: %d, regions: %d" % (s_count, stride, window_dim, r_count))

        # now we want to re_scale, instead of down_scaling the whole image, we down_scale the window
        # don't forget to recalculate the window area
        window_dim = int(window_dim * down_scale_factor)

    # run the detector on the regions
    start_time = time.clock()

    # after we collected all the regions from all the scales, send them to be detected
    scale_pred = __detect_from_scales_regions(recognition_model_path, detection_model_path, scale_regions, False)

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    scale_strong_regions = []

    print("... detection regions: %d, duration(min.): %f" % (r_count, duration))

    # construct the probability map for each scale and show it/ save it
    s_count = 0
    for pred, locations in zip(scale_pred, scale_locations):
        window_dim = scale_window_dim[s_count]
        s_count += 1
        map, weak_regions, strong_regions = __probability_map(img, pred, locations, window_dim, img_width, img_height, img_dim, s_count, False)
        if len(strong_regions) > 0:
            scale_strong_regions.append(strong_regions)
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, strong regions detected" % (s_count, stride, window_dim, r_count))
        else:
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, no regions detected" % (s_count, stride, window_dim, r_count))

    # now, after we finished scanning at all the levels, we should make the final verdict
    # by suppressing all the strong_regions that we extracted on different scales
    if len(scale_strong_regions) > 0:
        scale_regions = numpy.vstack(scale_strong_regions)
        __confidence_map(img, img_width, img_height, scale_regions, s_count)


def detect_from_file_fast(img_path, recognition_model_path, detection_model_path, img_dim):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    This function is called fast because it does not span the whole image but rather
    depends on the detection proposals/candidates
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    ##############################
    # Extract and detect regions #
    ##############################

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also down_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa

    # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
    # that means, we'll start with a window with initial size 320*320
    # for each ground_truth boundary, extract regions in such that:
    # 1. each region fully covers the boundary
    # 2. the boundary must not be smaller than the 1/5 of the region
    # ie. according to initial window size, the boundary must not be smaller than 80*80
    # but sure we will recognize smaller ground truth because we down_scale the window every step
    # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
    # don't forget that stride of sliding the window is dynamic

    # pre-process image by: equalize histogram and stretch intensity
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img = img.astype(float) / 255.0

    # min, max defines what is the range of scaling the sliding window
    # while scaling factor controls the amount of pixels to go down in each scale
    max_window_dim = int(img_dim * 2)
    min_window_dim = int(img_dim / 4)
    down_scale_factor = 0.9
    stride_factor = 0.1

    img_shape = img.shape
    img_width = img_shape[1]
    img_height = img_shape[0]

    s_count = 0
    r_count = 0

    # regions, locations and window_dim at each scale
    scale_regions = []
    scale_locations = []

    # we start by the window dimension = max and stop when it goes below
    # the min, at each iteration, we scale down the window dimension by a factor
    window_dim = max_window_dim

    # scale_down until you reach the min window
    # instead of scaling up the image itself, we scale down the sliding window
    while window_dim >= min_window_dim:

        # locations are the x,y position (top left) of the sliding-windows
        # regions are the extracted sliding windows from the image, passed
        # later to the detector to predict the location of the traffic sign with-in each one
        regions = []
        locations = []

        # stride is dynamic, smaller strides for smaller scales
        # this means that stride is equivialant to 2 pixels
        # when the window is resized to the img_dim (required for CNN)
        # r_factor = window_dim / min_window_dim
        # stride = int(stride_factor * int(r_factor))

        # simpler way to calculate the stride is: the stride is 10% of the current window dim
        stride = int(window_dim * stride_factor)

        s_count += 1
        r_count = 0

        # for the current scale of the window, extract regions, start from the
        y = 0
        x_count = 0
        y_count = 0
        region_shape = []

        # we only add windows that covers the strong detection proposals
        prop_max_dim = int(window_dim * 1.1)
        prop_min_dim = int(window_dim * 0.65)
        prop_weak, prop_strong, prop_map, prop_circles = CNN.prop.detection_proposal(img_color, min_dim=prop_min_dim, max_dim=prop_max_dim)
        if len(prop_strong) == 0:
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d" % (s_count, stride, window_dim, r_count))
            window_dim = int(window_dim * down_scale_factor)
            scale_regions.append([])
            scale_locations.append([])
            continue

        # loop on the detection proposals and take regions
        for proposal in prop_strong:
            x1 = proposal[0]
            y1 = proposal[1]
            x2 = proposal[2]
            y2 = proposal[3]
            region = img[y1:y2, x1:x2]
            region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))
            region = skimage.exposure.equalize_hist(region)

            # we only need to store the region, it's top-left corner and sliding window dim
            regions.append(region)
            locations.append([x1, y1, x2, y2])
            r_count += 1

        # append all the regions extracted from the current scale
        # we'll only do detection after we collect all the regions
        # from all the scales
        scale_regions.append(regions)
        scale_locations.append(locations)

        print("Scale: %d, stride: %d, window_dim: %d, regions: %d" % (s_count, stride, window_dim, r_count))

        # now we want to re_scale, instead of down_scaling the whole image, we down_scale the window
        # don't forget to recalculate the window area
        window_dim = int(window_dim * down_scale_factor)

    # run the detector on the regions
    start_time = time.clock()

    # after we collected all the regions from all the scales, send them to be detected
    scale_pred = __detect_from_scales_regions(recognition_model_path, detection_model_path, scale_regions)

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    scale_strong_regions = []

    print("... detection regions: %d, duration(min.): %f" % (r_count, duration))

    # construct the probability map for each scale and show it/ save it
    s_count = 0
    for pred, locations in zip(scale_pred, scale_locations):
        window_dim = scale_window_dim[s_count]
        s_count += 1
        map, weak_regions, strong_regions = __probability_map(img, pred, locations, window_dim, img_width, img_height, img_dim, s_count)
        if len(strong_regions) > 0:
            scale_strong_regions.append(strong_regions)
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, strong regions detected" % (s_count, stride, window_dim, r_count))
        else:
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, no regions detected" % (s_count, stride, window_dim, r_count))

    # now, after we finished scanning at all the levels, we should make the final verdict
    # by suppressing all the strong_regions that we extracted on different scales
    if len(scale_strong_regions) > 0:
        scale_regions = numpy.vstack(scale_strong_regions)
        __confidence_map(img, img_width, img_height, scale_regions, s_count)


def detect_from_file(img_path, recognition_model_path, detection_model_path, img_dim, proposals=True):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    ##############################
    # Extract and detect regions #
    ##############################

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also down_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa

    # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
    # that means, we'll start with a window with initial size 320*320
    # for each ground_truth boundary, extract regions in such that:
    # 1. each region fully covers the boundary
    # 2. the boundary must not be smaller than the 1/5 of the region
    # ie. according to initial window size, the boundary must not be smaller than 80*80
    # but sure we will recognize smaller ground truth because we down_scale the window every step
    # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
    # don't forget that stride of sliding the window is dynamic

    # pre-process image by: equalize histogram and stretch intensity
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # img_proc = skimage.exposure.equalize_hist(img)
    # img_proc = skimage.exposure.equalize_adapthist(img_proc, clip_limit=0.05, kernel_size=(8, 8))
    # img_proc = skimage.exposure.rescale_intensity(img_proc, in_range=(0.2, 0.75))
    img = img.astype(float) / 255.0

    # min, max defines what is the range of scaling the sliding window
    # while scaling factor controls the amount of pixels to go down in each scale
    max_window_dim = int(img_dim * 2)
    min_window_dim = int(img_dim / 4)
    down_scale_factor = 0.9
    stride_factor = 0.1

    img_shape = img.shape
    img_width = img_shape[1]
    img_height = img_shape[0]

    s_count = 0
    r_count = 0

    # regions, locations and window_dim at each scale
    scale_regions = []
    scale_locations = []
    scale_window_dim = []

    # we start by the window dimension = max and stop when it goes below
    # the min, at each iteration, we scale down the window dimension by a factor
    window_dim = max_window_dim

    # scale_down until you reach the min window
    # instead of scaling up the image itself, we scale down the sliding window
    while window_dim >= min_window_dim:

        # we need to save window_dim at each scale to resize back the predicted region
        scale_window_dim.append(window_dim)

        # locations are the x,y position (top left) of the sliding-windows
        # regions are the extracted sliding windows from the image, passed
        # later to the detector to predict the location of the traffic sign with-in each one
        regions = []
        locations = []

        # stride is dynamic, smaller strides for smaller scales
        # this means that stride is equivialant to 2 pixels
        # when the window is resized to the img_dim (required for CNN)
        # r_factor = window_dim / min_window_dim
        # stride = int(stride_factor * int(r_factor))

        # simpler way to calculate the stride is: the stride is 10% of the current window dim
        stride = int(window_dim * stride_factor)

        s_count += 1
        r_count = 0

        # for the current scale of the window, extract regions, start from the
        y = 0
        x_count = 0
        y_count = 0
        region_shape = []

        # check if option of using proposal is enabled
        if proposals:
            # important, instead of naively add every sliding window, we'll only add
            # windows that covers the strong detection proposals
            prop_max_dim = int(window_dim * 1.1)
            prop_min_dim = int(window_dim * 0.65)
            prop_weak, prop_strong, prop_map, prop_circles = CNN.prop.detection_proposal(img_color, min_dim=prop_min_dim, max_dim=prop_max_dim)
            if len(prop_strong) == 0:
                print("Scale: %d, stride: %d, window_dim: %d, regions: %d" % (s_count, stride, window_dim, r_count))
                window_dim = int(window_dim * down_scale_factor)
                scale_regions.append([])
                scale_locations.append([])
                continue

        while y <= img_height:
            x = 0
            x_count = 0
            while x <= img_width:

                # check if option of using proposal is enabled
                if proposals:
                    # if the current region does not intersect with the proposal, then ignore it
                    if not numpy.any(prop_map[y:y + window_dim, x:x + window_dim]):
                        x += stride
                        region_shape = (window_dim, window_dim)
                        continue

                # - add region to the region list
                # - adjust the position of the ground_truth to be relative to the window
                #   relative frame of reference (i.e not relative to the image)
                # - don't forget to re_scale the extracted/sampled region to be 28*28
                #   hence, multiply the relative position with this scaling accordingly
                # - also, the image needs to be preprocessed so it can be ready for the CNN
                region = img[y:y + window_dim, x:x + window_dim]
                region_shape = region.shape
                region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))

                # pre-process the region if needed
                region = skimage.exposure.equalize_hist(region)

                # we only need to store the region, it's top-left corner and sliding window dim
                regions.append(region)
                locations.append([x, y])

                r_count += 1

                # # # save region for experiemnt
                # filePathWrite = "D:\\_Dataset\\GTSDB\\Test_Regions\\%s_%s.png" % ("{0:03d}".format(s_count), "{0:03d}".format(r_count))
                # img_save = region * 255
                # img_save = img_save.astype(int)
                # cv2.imwrite(filePathWrite, img_save)

                x_count += 1
                x += stride
                if region_shape[1] < window_dim:
                    break

            y_count += 1
            if region_shape[0] < window_dim:
                break
            y += stride

        # append all the regions extracted from the current scale
        # we'll only do detection after we collect all the regions
        # from all the scales
        scale_regions.append(regions)
        scale_locations.append(locations)

        print("Scale: %d, stride: %d, window_dim: %d, regions: %d" % (s_count, stride, window_dim, r_count))

        # now we want to re_scale, instead of down_scaling the whole image, we down_scale the window
        # don't forget to recalculate the window area
        window_dim = int(window_dim * down_scale_factor)

    # run the detector on the regions
    start_time = time.clock()

    # after we collected all the regions from all the scales, send them to be detected
    scale_pred = __detect_from_scales_regions(recognition_model_path, detection_model_path, scale_regions)

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    scale_strong_regions = []

    print("... detection regions: %d, duration(min.): %f" % (r_count, duration))

    # construct the probability map for each scale and show it/ save it
    s_count = 0
    for pred, locations in zip(scale_pred, scale_locations):
        window_dim = scale_window_dim[s_count]
        s_count += 1
        map, weak_regions, strong_regions = __probability_map(img, pred, locations, window_dim, img_width, img_height, img_dim, s_count)
        if len(strong_regions) > 0:
            scale_strong_regions.append(strong_regions)
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, strong regions detected" % (s_count, stride, window_dim, r_count))
        else:
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, no regions detected" % (s_count, stride, window_dim, r_count))

    # now, after we finished scanning at all the levels, we should make the final verdict
    # by suppressing all the strong_regions that we extracted on different scales
    if len(scale_strong_regions) > 0:
        scale_regions = numpy.vstack(scale_strong_regions)
        __confidence_map(img, img_width, img_height, scale_regions, s_count)


def detect_from_file_slow_1(img_path, recognition_model_path, detection_model_path, pre_process=True, proposals=True):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    ##############################
    # Build the detector         #
    ##############################

    loaded_objects = CNN.utils.load_model(model_path=recognition_model_path, model_type=CNN.enums.ModelType._02_conv3_mlp2)
    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]

    # since we don't know that batch size in advance, let's say 500
    # and whatever regions we extract from the image we're going to split
    # them to batches and if the remainder is not zero, we're going to zero-pad
    # the remainder. For example the batch size is 500 and we've 600 regions
    # then pad the regions to be 1000 images and split into 2 patches
    batch_size = 1000

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    layer0_input = T.tensor4(name='input')
    layer0_img_dim = img_dim
    layer0_img_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)
    layer3_input_shape = (batch_size, nkerns[2] * layer3_img_dim * layer3_img_dim)

    # layer 0, 1, 2: Conv-Pool
    layer0_output = CNN.conv.convpool_layer(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        pool_size=pool_size
    )
    layer1_output = CNN.conv.convpool_layer(
        input=layer0_output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        pool_size=pool_size
    )
    layer2_output = CNN.conv.convpool_layer(
        input=layer1_output, W=layer2_W, b=layer2_b,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        pool_size=pool_size
    )
    # do the filtering using 3 layers of Conv+Pool
    conv_fn = theano.function([layer0_input], layer2_output)

    # load the regression model
    with open(detection_model_path, 'rb') as f:
        nn_regression = pickle.load(f)

    ##############################
    # Extract and detect regions #
    ##############################

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also down_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa

    # pre-process image by: equalize histogram and stretch intensity
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_proc = skimage.exposure.equalize_hist(img)
    img_proc = skimage.exposure.rescale_intensity(img_proc, in_range=(0.2, 0.75))
    img = img.astype(float) / 255.0

    # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
    # that means, we'll start with a window with initial size 320*320
    # for each ground_truth boundary, extract regions in such that:
    # 1. each region fully covers the boundary
    # 2. the boundary must not be smaller than the 1/5 of the region
    # ie. according to initial window size, the boundary must not be smaller than 80*80
    # but sure we will recognize smaller ground truth because we down_scale the window every step
    # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
    # don't forget that stride of sliding the window is dynamic

    down_scale_factor = 0.9
    window_dim = 160
    stride_factor = 10
    img_shape = img.shape
    img_width = img_shape[1]
    img_height = img_shape[0]

    s_count = 0

    # regions predicted from the previous scale
    weak_regions = []
    strong_regions = []
    scale_regions = []

    # scale_down until you reach the min window
    # instead of scaling up the image itself, we scale down the sliding window
    while window_dim >= img_dim:

        # locations are the x,y position (top left) of the sliding-windows
        # regions are the extracted sliding windows from the image, passed
        # later to the detector to predict the location of the traffic sign with-in each one
        regions = []
        locations = []

        # stride is dynamic, smaller strides for smaller scales
        # this means that stride is equivialant to 2 pixels
        # when the window is resized to the img_dim (required for CNN)
        r_factor = window_dim / img_dim
        stride = int(stride_factor * int(r_factor))

        s_count += 1
        r_count = 0

        # for the current scale of the window, extract regions, start from the
        y = 0
        x_count = 0
        y_count = 0
        region_shape = []

        # check if option of using proposal is enabled
        if proposals:
            # important, instead of naively add every sliding window, we'll only add
            # windows that covers the strong detection proposals
            proposals_weak, proposals_strong, proposals_map, prop_circles = CNN.prop.detection_proposal(img_proc, min_dim=int(window_dim / 2), max_dim=window_dim)
            if len(proposals_strong) == 0:
                print("Scale: %d, stride: %d, window_dim: %d, regions: %d, duration(min.): %f" % (s_count, stride, window_dim, r_count, 0))
                window_dim = int(window_dim * down_scale_factor)
                continue

        while y <= img_height:
            x = 0
            x_count = 0
            while x <= img_width:

                # check if option of using proposal is enabled
                if proposals:
                    # if the current region does not intersect with the proposal, then ignore it
                    if not numpy.any(proposals_map[y:y + window_dim, x:x + window_dim]):
                        x += stride
                        region_shape = (window_dim, window_dim)
                        continue

                # - add region to the region list
                # - adjust the position of the ground_truth to be relative to the window
                #   relative frame of reference (i.e not relative to the image)
                # - don't forget to re_scale the extracted/sampled region to be 28*28
                #   hence, multiply the relative position with this scaling accordingly
                # - also, the image needs to be preprocessed so it can be ready for the CNN
                region = img[y:y + window_dim, x:x + window_dim]
                region_shape = region.shape
                region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))

                # pre-process the region if needed
                if pre_process:
                    region = skimage.exposure.equalize_hist(region)

                # we only need to store the region, it's top-left corner and sliding window dim
                regions.append(region)
                locations.append([x, y])

                r_count += 1

                # # # save region for experiemnt
                # filePathWrite = "D:\\_Dataset\\GTSDB\\Test_Regions\\%s_%s.png" % ("{0:03d}".format(s_count), "{0:03d}".format(r_count))
                # img_save = region * 255
                # img_save = img_save.astype(int)
                # cv2.imwrite(filePathWrite, img_save)

                x_count += 1
                x += stride
                if region_shape[1] < window_dim:
                    break

            y_count += 1
            if region_shape[0] < window_dim:
                break
            y += stride

        # now we want to re_scale, instead of down_scaling the whole image, we down_scale the window
        # don't forget to recalculate the window area
        window_dim = int(window_dim * down_scale_factor)

        # split it to batches first, zero-pad them if needed
        regions = numpy.asarray(regions)
        n_regions = regions.shape[0]
        if n_regions % batch_size != 0:
            n_remaining = batch_size - (n_regions % batch_size)
            regions_padding = numpy.zeros(shape=(n_remaining, img_dim, img_dim), dtype=float)
            regions = numpy.vstack((regions, regions_padding))

        # run the detector on the regions
        start_time = time.clock()

        # loop on the batches of the regions
        n_batches = int(regions.shape[0] / batch_size)
        pred = []
        for i in range(n_batches):
            t1 = time.clock()
            # prediction: CNN filtering then MLP regression
            batch = regions[i * batch_size: (i + 1) * batch_size]
            batch = batch.reshape(layer0_img_shape)
            filters = conv_fn(batch)
            filters = filters.reshape(layer3_input_shape).astype("float32")
            batch_pred = nn_regression.predict(filters)
            pred.append(batch_pred)
            t2 = time.clock()
            print("... batch: %i/%i, time(sec.): %f" % ((i + 1), n_batches, t2 - t1))

        # after getting all the predictions, remove the padding
        pred = numpy.vstack(pred)
        pred = pred[0:n_regions]

        # scale-back the the predicted values to it's original scale
        pred = numpy.rint(((pred * img_dim) + img_dim) / 2).astype(int)
        pred[pred > img_dim - 1] = img_dim - 1
        pred[pred < 0] = 0

        end_time = time.clock()
        duration = (end_time - start_time) / 60

        # after getting the predictions, construct the probability map and show it/ save it
        # since we're working on course-to-fine fashion, so for the next scale,
        # we'll only explore the regions detected in the current scale
        map, weak_regions, strong_regions = __probability_map(img, pred, locations, window_dim, img_width, img_height, img_dim, s_count)
        if len(strong_regions) > 0:
            scale_regions.append(strong_regions)
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, duration(min.): %f" % (s_count, stride, window_dim, r_count, duration))
        else:
            print("Scale: %d, stride: %d, window_dim: %d, regions: %d, no regions detected" % (s_count, stride, window_dim, r_count))

    # now, after we finished scanning at all the levels, we should make the final verdict
    # by suppressing all the strong_regions that we extracted on different scales
    if len(scale_regions) > 0:
        scale_regions = numpy.vstack(scale_regions)
        __confidence_map(img, img_width, img_height, scale_regions, s_count)


def detect_from_file_slow_2(img_path, recognition_model_path, detection_model_path, img_dim, model_type=CNN.enums.ModelType, classifier=CNN.enums.ClassifierType.logit):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also down_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(float) / 255.0

    # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
    # that means, we'll start with a window with initial size 320*320
    # for each ground_truth boundary, extract regions in such that:
    # 1. each region fully covers the boundary
    # 2. the boundary must not be smaller than the 1/5 of the region
    # ie. according to initial window size, the boundary must not be smaller than 80*80
    # but sure we will recognize smaller ground truth because we down_scale the window every step
    # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
    # don't forget that stride of sliding the window is dynamic

    down_scale_factor = 0.9
    window_dim = 120
    stride_factor = 10
    img_shape = img.shape
    img_width = img_shape[1]
    img_height = img_shape[0]

    s_count = 0

    # scale_down until you reach the min window
    # instead of scaling up the image itself, we scale down the sliding window
    while window_dim >= img_dim:

        # locations are the x,y position (top left) of the sliding-windows
        # regions are the extracted sliding windows from the image, passed
        # later to the detector to predict the location of the traffic sign with-in each one
        regions = []
        locations = []

        # stride is dynamic, smaller strides for smaller scales
        # this means that stride is equivialant to 2 pixels
        # when the window is resized to the img_dim (required for CNN)
        r_factor = window_dim / img_dim
        stride = int(stride_factor * int(r_factor))

        s_count += 1
        r_count = 0

        # for the current scale of the window, extract regions, start from the
        # y_range = numpy.arange(start=0, stop=img_height, step=stride, dtype=int).tolist()
        # x_range = numpy.arange(start=0, stop=img_width, step=stride, dtype=int).tolist()
        y = 0
        x_count = 0
        y_count = 0
        region_shape = []
        while y <= img_height:
            x = 0
            x_count = 0
            while x <= img_width:
                # - add region to the region list
                # - adjust the position of the ground_truth to be relative to the window
                #   relative frame of reference (i.e not relative to the image)
                # - don't forget to re_scale the extracted/sampled region to be 28*28
                #   hence, multiply the relative position with this scaling accordingly
                # - also, the image needs to be preprocessed so it can be ready for the CNN
                region = img[y:y + window_dim, x:x + window_dim]
                region_shape = region.shape
                region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))
                # we only need to store the region, it's top-left corner and sliding window dim
                regions.append(region)
                locations.append((x, y))

                r_count += 1

                # save region for experiemnt
                # filePathWrite = "D:\\_Dataset\\GTSDB\\Test_Regions\\%s_%s.png" % ("{0:03d}".format(s_count), "{0:03d}".format(r_count))
                # img_save = region * 255
                # img_save = img_save.astype(int)
                # cv2.imwrite(filePathWrite, img_save)

                x_count += 1
                x += stride
                if region_shape[1] < window_dim:
                    break

            y_count += 1
            if region_shape[0] < window_dim:
                break
            y += stride

        # now we want to re_scale, instead of down_scaling the whole image, we down_scale the window
        # don't forget to recalculate the window area
        window_dim = int(window_dim * down_scale_factor)

        # send the regions for the detector and convert the result to the probability map
        batch = numpy.asarray(regions)
        d_pred, d_duration = __detect_batch(batch, recognition_model_path, detection_model_path, model_type, classifier)

        # now, after getting the predictions, construct the probability map and show it
        map = __probability_map(d_pred, locations, window_dim, img_width, img_height, img_dim, s_count)
        print("Scale: %d, stride: %d, window_dim: %d, regions: %d" % (s_count, stride, window_dim, r_count))

    x = 10


def __detect_from_scales_regions(recognition_model_path, detection_model_path, scale_regions, regression=True):
    # stack the regions of all the scales in one array
    # please note that a scale can have no regions, so using vstack wouldn't work
    # remove the scales with empty regions then use vstack
    if regression:
        regions = []
        for r in scale_regions:
            if len(r) > 0:
                regions.append(r)
        regions = numpy.vstack(regions)
    else:
        regions = scale_regions
    batch_size = len(regions)

    ##############################
    # Build the detector         #
    ##############################

    loaded_objects = CNN.utils.load_model(model_path=recognition_model_path, model_type=CNN.enums.ModelType._02_conv3_mlp2)
    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    layer0_input = T.tensor4(name='input')
    layer0_img_dim = img_dim
    layer0_img_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)
    layer3_input_shape = (batch_size, nkerns[2] * layer3_img_dim * layer3_img_dim)

    # layer 0, 1, 2: Conv-Pool
    layer0_output = CNN.conv.convpool_layer(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        pool_size=pool_size
    )
    layer1_output = CNN.conv.convpool_layer(
        input=layer0_output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        pool_size=pool_size
    )
    layer2_output = CNN.conv.convpool_layer(
        input=layer1_output, W=layer2_W, b=layer2_b,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        pool_size=pool_size
    )
    # do the filtering using 3 layers of Conv+Pool
    conv_fn = theano.function([layer0_input], layer2_output)

    # load the regression model
    with open(detection_model_path, 'rb') as f:
        nn_regression = pickle.load(f)

    ##############################
    # Start detection            #
    ##############################

    t1 = time.clock()
    # prediction: CNN filtering then MLP regression
    batch = regions.reshape(layer0_img_shape)
    filters = conv_fn(batch)
    filters = filters.reshape(layer3_input_shape).astype("float32")
    pred = nn_regression.predict(filters)
    t2 = time.clock()
    print("... prediction time(sec.): %f" % (t2 - t1))

    # in case of regression
    if regression:

        # scale-back the the predicted values to it's original scale
        pred = numpy.rint(((pred * img_dim) + img_dim) / 2).astype(int)
        pred[pred > img_dim - 1] = img_dim - 1
        pred[pred < 0] = 0

        # split the predictions to their scales
        # i.e. re-arrange the pred to scale_pred
        scale_pred = []
        r_count = 0
        for r in scale_regions:
            n = len(r)
            scale_pred.append(pred[r_count:r_count + n])
            r_count += n
    else:
        scale_pred = pred

    return scale_pred


def __detect_batch(batch, recognition_model_path, detection_model_path, model_type=CNN.enums.ModelType, classifier=CNN.enums.ClassifierType.logit):
    if model_type == CNN.enums.ModelType._01_conv2_mlp2:
        # return __detect_batch_shallow_model(batch, model_path, classifier)
        return None
    elif model_type == CNN.enums.ModelType._02_conv3_mlp2:
        return __detect_batch_deep_model(batch, recognition_model_path, detection_model_path, classifier)
    else:
        raise Exception("Unknown model type")


def __detect_batch_deep_model(batch, recognition_model_path, detection_model_path, classifier=CNN.enums.ClassifierType.logit):
    loaded_objects = CNN.utils.load_model(model_path=recognition_model_path, model_type=CNN.enums.ModelType._02_conv3_mlp2)

    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]
    batch_size = batch.shape[0]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    layer0_input = T.tensor4(name='input')
    layer0_img_dim = img_dim
    layer0_img_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)
    layer3_input_shape = (batch_size, nkerns[2] * layer3_img_dim * layer3_img_dim)

    # layer 0, 1, 2: Conv-Pool
    layer0_output = CNN.conv.convpool_layer(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        pool_size=pool_size
    )
    layer1_output = CNN.conv.convpool_layer(
        input=layer0_output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        pool_size=pool_size
    )
    layer2_output = CNN.conv.convpool_layer(
        input=layer1_output, W=layer2_W, b=layer2_b,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        pool_size=pool_size
    )
    # do the filtering using 3 layers of Conv+Pool
    conv_fn = theano.function([layer0_input], layer2_output)
    with open(detection_model_path, 'rb') as f:
        nn_regression = pickle.load(f)

    start_time = time.clock()

    # prediction
    batch = batch.reshape(layer0_img_shape)
    filters = conv_fn(batch)
    filters = filters.reshape(layer3_input_shape).astype("float32")
    d_pred = nn_regression.predict(filters)

    # scale-back the the predicted values to it's original scale
    d_pred = numpy.rint(((d_pred * img_dim) + img_dim) / 2).astype(int)
    d_pred[d_pred > img_dim - 1] = img_dim - 1
    d_pred[d_pred < 0] = 0

    end_time = time.clock()
    d_duration = end_time - start_time

    return d_pred, d_duration


def __detect_img(img4D, model_path, model_type=CNN.enums.ModelType, classifier=CNN.enums.ClassifierType.logit):
    if model_type == CNN.enums.ModelType._01_conv2_mlp2:
        return __detect_img_shallow_model(img4D, model_path, classifier)
    elif model_type == CNN.enums.ModelType._02_conv3_mlp2:
        return __detect_img_deep_model(img4D, model_path, classifier)
    else:
        raise Exception("Unknown model type")


def __detect_img_shallow_model(img4D, model_path, classifier=CNN.enums.ClassifierType.logit):
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


def __detect_img_deep_model(img4D, model_path, classifier=CNN.enums.ClassifierType.logit):
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


# endregion

# region Helping Functions


def __probability_map(img, predictions, locations, window_dim, img_width, img_height, img_dim, count, regression=True, **kwargs):
    # parameters of the algorithm
    min_dim = img_dim / 2
    overlap_thresh = 0.4
    min_overlap = 5

    if 'overlap_thresh' in kwargs:
        overlap_thresh = kwargs['overlap_thresh']
    if 'min_overlap' in kwargs:
        min_overlap = kwargs['min_overlap']

    r_factor = window_dim / img_dim
    locations = numpy.asarray(locations)
    predictions = numpy.asarray(predictions)
    n = predictions.shape[0]

    # create an image
    map = numpy.zeros(shape=(img_height, img_width))
    new_regions = []
    for i in range(0, n):

        # in case of regression, mark only the predicted region
        # else, mark the whole sliding window
        if regression:
            x1 = predictions[i, 0]
            x2 = predictions[i, 2]
            y2 = predictions[i, 3]
            y1 = predictions[i, 1]
            if (x2 - x1) < min_dim or (y2 - y1) < min_dim:
                continue
            new_region = numpy.rint(predictions[i] * r_factor)
        else:
            if predictions[i]:
                new_region = [0, 0, window_dim, window_dim]
            else:
                continue

        location = locations[i]
        x1 = int(new_region[0] + location[0])
        y1 = int(new_region[1] + location[1])
        x2 = int(new_region[2] + location[0])
        y2 = int(new_region[3] + location[1])
        new_regions.append([x1, y1, x2, y2])
        map[y1:y2, x1:x2] += 1

    # check if no region found
    if len(new_regions) == 0:
        map = []
        weak_regions = []
        strong_regions = []
        return map, weak_regions, strong_regions

    # suppress the new regions and raw them with red color
    weak_regions, strong_regions = CNN.nms.suppression(new_regions, overlap_thresh, min_overlap)

    # normalize image before drawing
    map = map * 255 / (map.max() - map.min())
    img_normalized = img * 255

    # we also may want to blend the image and the probability map
    blend_value = 0.25
    map_blend = cv2.addWeighted(img_normalized, blend_value, map, 1 - blend_value, 0.0)
    map_blend = map_blend.astype(int)

    # convert to RGB before drawing colored boxes
    map_color = numpy.zeros(shape=(img_height, img_width, 3))
    for i in range(3):
        map_color[:, :, i] = map_blend

    red_color = (0, 0, 255)
    yellow_color = (84, 212, 255)
    blue_color = (255, 0, 0)
    for loc in weak_regions:
        cv2.rectangle(map_color, (loc[0], loc[1]), (loc[2], loc[3]), yellow_color, 1)
    for loc in strong_regions:
        cv2.rectangle(map_color, (loc[0], loc[1]), (loc[2], loc[3]), red_color, 2)

    cv2.imwrite("D:\\_Dataset\\GTSDB\\Test_Regions\\" + "{0:05d}.png".format(count), map_color)

    # return the map to be exploited later by the detector, for the next scale
    return map, weak_regions, strong_regions


def __confidence_map(img, img_width, img_height, scale_regions, scale_count, **kwargs):
    overlap_thresh = 0.5
    min_overlap = 3
    save_img = True

    if 'overlap_thresh' in kwargs:
        overlap_thresh = kwargs['overlap_thresh']
    if 'min_overlap' in kwargs:
        min_overlap = kwargs['min_overlap']

    weak_regions, strong_regions = CNN.nms.suppression(scale_regions, overlap_thresh, min_overlap)

    # normalize the image
    img_normalized = img * 255

    # convert to RGB before drawing colored boxes
    map_color = numpy.zeros(shape=(img_height, img_width, 3))
    for i in range(3):
        map_color[:, :, i] = img_normalized

    red_color = (0, 0, 255)
    yellow_color = (84, 212, 255)
    blue_color = (255, 0, 0)
    for loc in weak_regions:
        cv2.rectangle(map_color, (loc[0], loc[1]), (loc[2], loc[3]), yellow_color, 1)
    for loc in strong_regions:
        cv2.rectangle(map_color, (loc[0], loc[1]), (loc[2], loc[3]), red_color, 2)

    cv2.imwrite("D:\\_Dataset\\GTSDB\\Test_Regions\\" + "{0:05d}.png".format(scale_count + 1), map_color)

# endregion
