__author__ = 'Noureldien'

import gzip
import pickle
import numpy
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def run():

    f = gzip.open('mnist.pkl.gz', 'rb')

    # two methods to load
    # method 1
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()

    # method 2
    #train_set, valid_set, test_set = pickle.load(f)

    f.close()

    train_x, train_y = train_set

    plt.imshow(train_x[2].reshape((28, 28)), cmap = cm.Greys_r)
    plt.show()


















