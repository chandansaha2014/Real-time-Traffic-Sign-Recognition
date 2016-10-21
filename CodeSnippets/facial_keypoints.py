import csv
import os
import numpy
import pickle
from datetime import datetime
import os
import sys
from matplotlib import pyplot
import numpy as np
from lasagne import layers
import lasagne.layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

def serialize_data():
    images = []
    points = []
    file_path = 'D:/_Dataset/KFKD/kfkd_training.csv'
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if reader.line_num % 100 == 1:
                print(reader.line_num)
            if reader.line_num == 1:
                continue
            for r in range(0, 29):
                if row[r] == "":
                    row[r] = "0.0"
            img = [int(s) for s in row[30].split(' ')]
            pts = [float(s) for s in row[0:29]]
            images.append(img)
            points.append(pts)

    n = len(points)
    n_train = n * 3 / 4
    n_valid = n * (1 / 4) * (4 / 5)

    images = numpy.asarray(images, dtype=float) / 255.0
    points = numpy.asarray(points, dtype=int)

    train_imgs = images[0:n_train]
    valid_imgs = images[n_train:n_train + n_valid]
    test_imgs = images[n_train + n_valid: n]

    train_points = points[0:n_train]
    valid_points = points[n_train:n_train + n_valid]
    test_points = points[n_train + n_valid: n]

    data = ((train_imgs, train_points), (valid_imgs, valid_points), (test_imgs, test_points))
    pickle.dump(data, open("D:/_Dataset/KFKD/kfkd.pkl", "wb"))

    print(n_train)
    print(n_valid)
    print(n - n_train - n_valid)

    x = 10

def float32(k):
    return np.cast['float32'](k)

def train():
    data = pickle.load(open("D:/_Dataset/KFKD/kfkd.pkl", "rb"))
    x = 10

    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', lasagne.layers.Conv2DLayer),
            ('pool1', lasagne.layers.MaxPool2DLayer),
            ('dropout1', lasagne.layers.DropoutLayer),
            ('conv2', lasagne.layers.Conv2DLayer),
            ('pool2', lasagne.layers.MaxPool2DLayer),
            ('dropout2', lasagne.layers.layers.DropoutLayer),
            ('conv3', lasagne.layers.Conv2DLayer),
            ('pool3', lasagne.layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=1000,
        dropout4_p=0.5,
        hidden5_num_units=1000,
        output_num_units=30, output_nonlinearity=None,

        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),

        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=200),
        ],
        max_epochs=3000,
        verbose=1,
    )
