import os
from os import listdir
from os.path import isfile, join
import time
import csv
import pickle
import gzip
import numpy
import theano
import theano.tensor
import PIL
import PIL.Image
import skimage
import skimage.transform
import skimage.exposure
import cv2

import CNN
import CNN.recog
import CNN.enums
import CNN.consts
import CNN.conv

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import random

# region Misc

def numpy_to_string(arr):
    arr *= 100
    return ", ".join(format(x, ".0f") + "%" for x in arr.tolist())


def float32(k):
    return numpy.cast['float32'](k)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    # ############################# Batch iterator ###############################
    # This is just a simple helper function iterating over training data in
    # mini-batches of a particular size, optionally in random order. It assumes
    # data is available as numpy arrays. For big datasets, you could load numpy
    # arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
    # own custom data iteration function. For small datasets, you can also copy
    # them to GPU at once for slightly improved performance. This would involve
    # several changes in the main program, though, and is not demonstrated here.
    :param inputs:
    :param targets:
    :param batchsize:
    :param shuffle:
    :return:
    """

    assert len(inputs) == len(targets)
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# endregion

# region nolearn helping classes


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001, count=None):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
        self.count = count

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = numpy.linspace(self.start, self.stop, nn.max_epochs)
        elif self.count is not None:
            self.ls = numpy.linspace(self.start, self.stop, self.count)

        epoch = train_history[-1]['epoch']
        # this is hot-fix to fix bug when resuming training model
        # that was serialized before
        n_ls = self.ls.shape[0]
        if epoch > n_ls:
            epoch = epoch % n_ls

        new_value = numpy.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = numpy.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


# endregion

# region Load Model/Data

def unzip_load_data(dataset):
    f = gzip.open(dataset, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    del f

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def load_model(model_path, model_type=CNN.enums.ModelType._01_conv2_mlp2):
    n = 0
    if model_type == CNN.enums.ModelType._01_conv2_mlp2:
        n = 14
    elif model_type == CNN.enums.ModelType._02_conv3_mlp2:
        n = 16
    else:
        raise Exception("Unknown model type")
    save_file = open(model_path, 'rb')
    loaded_objects = []
    for i in range(n):
        loaded_objects.append(pickle.load(save_file))
    save_file.close()
    return loaded_objects


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy

    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    shared_y_casted = theano.tensor.cast(shared_y, 'int32')
    return shared_x, shared_y_casted


# endregion

# region Pre-process

def preprocess_dataset_train(img_dim, superclass_type=CNN.enums.SuperclassType._01_Prohibitory):
    directory1 = "D:\\_Dataset\\GTSRB\\Final_Training_Cropped\\"
    directory2 = "D:\\_Dataset\\GTSRB\\Final_Training_Preprocessed_%d\\" % (img_dim)

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        classes = CNN.consts.ClassesIDs.PROHIB_CLASSES
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        classes = CNN.consts.ClassesIDs.WARNING_CLASSES
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        classes = CNN.consts.ClassesIDs.MANDATORY_CLASSES
    else:
        raise Exception("Sorry, un-recognized super-class type")

    for class_id in classes:
        folderName = "{0:05d}\\".format(class_id)
        subDirectory1 = directory1 + folderName
        subDirectory2 = directory2 + folderName
        files = [f for f in listdir(subDirectory1) if isfile(join(subDirectory1, f))]
        # create directory to save files in
        if not os.path.exists(subDirectory2):
            os.mkdir(subDirectory2)
        for file in files:
            # do the following steps : read -> Grayscale -> imadjust -> histeq
            # -> adapthisteq -> ContrastStretchNorm -> resize -> write
            filePathRead = join(subDirectory1, file)
            filePathWrite = join(subDirectory2, file)
            preprocess_image_(filePathRead=filePathRead, filePathWrite=filePathWrite, resize_dim=img_dim)

        print('Finish Class: ' + folderName)


def preprocess_dataset_test(img_dim):
    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test_PNG\\GT-final_test.annotated.csv"
    test_images = []
    test_classes = []
    # get the ground truth of the test data
    with open(csvFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if row[7] != "ClassId":
                test_images.append(row[0][:-4])
                test_classes.append(int(row[7]))

    directory1 = "D:\\_Dataset\\GTSRB\\Final_Test_Cropped\\"
    directory2 = "D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed_%d\\" % (img_dim)

    # for the time being, only process the prohibitory traffic signs
    prohibitory_classes = CNN.consts.ClassesIDs.PROHIB_CLASSES
    processed_classes = []

    files = [f for f in listdir(directory1) if isfile(join(directory1, f))]
    count = 0
    for file in files:
        # get class_id of the current image
        index = test_images.index(file[:-4])
        class_id = test_classes[index]
        if class_id not in prohibitory_classes:
            continue
        count += 1
        if count % 500 == 1:
            print("Finish %d images" % (count))
        # do the following steps : read -> Grayscale -> imadjust -> histeq
        # -> adapthisteq -> ContrastStretchNorm -> resize -> write
        filePathRead = join(directory1, file)
        filePathWrite = join(directory2, file)
        is_processed = preprocess_image_(filePathRead=filePathRead, filePathWrite=filePathWrite, resize_dim=img_dim)
        if is_processed:
            processed_classes.append(class_id)

    pickle.dump(processed_classes, open("D:\\_Dataset\\GTSRB\\gtsrb_prohibitroy_classes.pkl", 'wb'))


def preprocess_image_(filePathRead, filePathWrite, resize_dim=0):
    img = cv2.imread(filePathRead)
    # if img.shape[0] < 40 or img.shape[1] < 40:
    #    return False
    img = preprocess_image(img, resize_dim)
    img *= 255
    img = img.astype(int)
    cv2.imwrite(filePathWrite, img)
    return True


def preprocess_image(img, resize_dim=0):
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = skimage.exposure.equalize_hist(img_gs)
    img_adeq = skimage.exposure.equalize_adapthist(img_eq, clip_limit=0.2, kernel_size=(8, 8))
    img_int = skimage.exposure.rescale_intensity(img_adeq, in_range=(0.1, 0.8))
    if resize_dim > 0:
        img_int = skimage.transform.resize(img_int, output_shape=(resize_dim, resize_dim))
    return img_int

    plot_images = False

    if plot_images:
        # region Plot results
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 5))
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img, axes[:, 0])
        ax_img.set_title('Low contrast image')
        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_ylabel('Number of pixels')
        ax_hist.set_yticks(numpy.linspace(0, y_max, 5))
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img_eq, axes[:, 1])
        ax_img.set_title('Histogram equalization')
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img_adeq, axes[:, 2])
        ax_img.set_title('Adaptive equalization')
        ax_img, ax_hist, ax_cdf = __plot_img_and_hist(img_int, axes[:, 3])
        ax_img.set_title('Contrast stretching')
        ax_cdf.set_ylabel('Fraction of total intensity')
        ax_cdf.set_yticks(numpy.linspace(0, 1, 5))
        # prevent overlap of y-axis labels
        fig.subplots_adjust(wspace=0.4)
        plt.show()

        return
        # endregion


def probability_map(img_path, model_path, classifier=CNN.enums.ClassifierType.logit, window_size=28):
    """
    For the given image, apply sliding window algorithm over different scales and return the heat-probability map
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    img = cv2.imread(img_path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scaling_factor = 0.75
    stride = int(window_size * 0.5)

    scales = -1
    d = min(img.shape[0], img.shape[1])
    while d >= window_size:
        d = int(d * scaling_factor)
        scales += 1

    batch_counts = []
    batch = []
    shapes = []

    # loop on different scales of the image
    for s in range(0, scales):

        # split the images to small images using overlapping sliding window
        # do classification of all of the small images as one batch
        # then output the classification result as probability map
        batch_count = 0
        y_range = numpy.arange(0, img_gs.shape[0] - window_size, stride)
        x_range = numpy.arange(0, img_gs.shape[1] - window_size, stride)
        for y in y_range:
            for x in x_range:
                roi = img_gs[y:y + window_size, x:x + window_size] / 255.0
                batch.append(roi.ravel())
                batch_count += 1

        batch_counts.append(batch_count)

        # re-scale the image to make it smaller
        shapes.append((len(y_range), len(x_range)))
        img_gs = skimage.transform.resize(img_gs, output_shape=(
            int(img_gs.shape[0] * scaling_factor), int(img_gs.shape[1] * scaling_factor)))

    # after we obained all the batches for all the image scales, classify the batches
    batch_np = numpy.asarray(batch, float)
    c_result, c_prob, c_duration = CNN.recog.classify_batch(batch_np, model_path, classifier)
    print('Classification of image batches in %f sec.' % (c_duration))

    # get the classification result and probability each scale
    return
    offset = 0
    c_prob[c_prob < 0.75] = 0
    for i in range(0, len(batch_counts)):
        results = c_result[offset: offset + batch_counts[i]]
        p_maps = numpy.asarray(c_prob[offset: offset + batch_counts[i]])
        p_maps = p_maps.reshape(p_maps.shape[1], p_maps.shape[0])
        __plot_prob_maps(p_maps, shapes[i], i + 1)
        # for p_map in p_maps:
        #    p_map = p_map.reshape(shapes[i])
        #    # display map
        offset += batch_counts[i]

        # generate image to show confidence maps for the 3 classes, each with probabilities
        # note that for each class, we have confidence map at each scale


def rgb_to_gs(path):
    img = cv2.imread(path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img_gs


def rgb_to_gs_and_save(path):
    img = cv2.imread(path)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img_save = img_gs * 255
    img_save = img_save.astype(int)
    cv2.imwrite(path, img_save)


def ppm_to_png():
    from os import listdir
    from os.path import isfile, join

    csvFileName = "D:\\_Dataset\\GTSDB\\Train\\gt.txt"

    directory1 = "D:\\_Dataset\\GTSDB\\Train\\00\\"
    directory2 = "D:\\_Dataset\\GTSDB\\Train_PNG\\"

    for i in range(0, 4):
        file1 = "{0:05d}.ppm".format(i)
        file2 = "{0:05d}.png".format(i)
        filePathRead = join(directory1, file1)
        filePathWrite = join(directory1, file2)
        img = cv2.imread(filePathRead)
        cv2.imwrite(filePathWrite, img)


def __plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    """
    import skimage
    import skimage.exposure
    import matplotlib.pyplot as plt

    img = skimage.img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = skimage.exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def __plot_prob_maps(maps, shape, fig_num):
    plt.figure(fig_num)
    plt.gray()
    plt.ion()
    length = len(maps)
    for i in range(0, length):
        plt.subplot(1, length, i + 1)
        plt.axis('off')
        p_map = maps[i]
        p_map = p_map.reshape(shape)
        plt.imshow(p_map)
    plt.show()


# endregion

# region GTSR

def serialize_gtsr_test():
    data_path = "D:\\_Dataset\GTSRB\\gtsrb_serialized_test_28.pkl"
    directoryTest = "D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed"
    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test_PNG\\GT-final_test.annotated.csv"

    img_dim = 28
    test_names = []
    test_classes = []
    test_images = []

    # get the ground truth of the test data
    with open(csvFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if row[7] == "ClassId":
                continue
            class_id = int(row[7])
            test_names.append(int(row[0][:-4]))
            test_classes.append(class_id)

    files = [f for f in listdir(directoryTest) if isfile(join(directoryTest, f))]
    for file in files:
        fileName = join(directoryTest, file)
        fileID = int(file[:-4])
        if not (fileID in test_names):
            continue
        img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        img = skimage.transform.resize(img, output_shape=(img_dim, img_dim))
        img = img.reshape(img_dim * img_dim, )
        test_images.append(img)

    test_images = numpy.asarray(test_images)
    test_classes = numpy.asarray(test_classes)
    data = (test_images, test_classes)

    print(test_images.shape)
    print(test_classes.shape)

    # this may cause memory problems
    pickle.dump(data, open(data_path, 'wb'))

    # p = pickle._Pickler(open(data_path, "wb"))
    # p.fast = True
    # p.dump(data)

    print("Finish Serializing Test Data")


def serialize_gtsr(img_dim, superclass_type, sampling=False):
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset

    :return:
    '''

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        classes_ids = CNN.consts.ClassesIDs.PROHIB_CLASSES
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        classes_ids = CNN.consts.ClassesIDs.WARNING_CLASSES
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        classes_ids = CNN.consts.ClassesIDs.MANDATORY_CLASSES
        type_char = 'm'
    elif superclass_type == CNN.enums.SuperclassType._04_Other:
        classes_ids = CNN.consts.ClassesIDs.OTHER_CLASSES
        type_char = 'o'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    train_images = []
    train_classes = []
    test_images = []
    test_classes = []
    all_test_names = []
    all_test_classes = []

    directoryTrain = "D:\\_Dataset\\GTSRB\\Final_Training_Preprocessed"
    directoryTest = "D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed"
    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test_PNG\\GT-final_test.annotated.csv"

    # get the ground truth of the test data
    with open(csvFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if row[7] == "ClassId":
                continue
            class_id = int(row[7])
            if class_id in classes_ids:
                all_test_names.append(int(row[0][:-4]))
                all_test_classes.append(class_id)

    # get the test data
    # smallest_dim = img_dim / 2
    smallest_dim = 0

    selected_test_names = []
    files = [f for f in listdir(directoryTest) if isfile(join(directoryTest, f))]
    for file in files:
        fileName = join(directoryTest, file)
        fileID = int(file[:-4])
        if not (fileID in all_test_names):
            continue
        img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        if img.shape[0] < smallest_dim or img.shape[1] < smallest_dim:
            continue
        img = skimage.transform.resize(img, output_shape=(img_dim, img_dim))
        img = img.reshape(img_dim * img_dim, )
        test_images.append(img)
        selected_test_names.append(int(file[:-4]))

    # now, loop on all the class_ids of the test
    # and only choose the class_ids of the images resized
    for name in selected_test_names:
        index = all_test_names.index(name)
        test_classes.append(all_test_classes[index])

    # get the training data
    sub_directories = [d for d in listdir(directoryTrain) if os.path.isdir(join(directoryTrain, d))]
    for d in sub_directories:
        directoryID = int(d)
        if not (int(d) in classes_ids):
            continue
        sub_directory = join(directoryTrain, d)
        onlyfiles = [f for f in listdir(sub_directory) if isfile(join(sub_directory, f))]

        # we don't want to take all the images in the class, we want only small sample
        n_files = len(onlyfiles)
        idx = numpy.arange(start=0, stop=n_files, dtype=int).tolist()
        if sampling:
            random.shuffle(idx)
            n_samples = 500
            if n_files < n_samples:
                n_samples = n_files
        else:
            n_samples = n_files
        for f_index in idx[0:n_samples]:
            file = onlyfiles[f_index]
            fileName = join(sub_directory, file)
            img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            if img.shape[0] < smallest_dim or img.shape[1] < smallest_dim:
                continue
            img = skimage.transform.resize(img, output_shape=(img_dim, img_dim))
            img = img.reshape(img_dim * img_dim, )
            train_images.append(img)
            train_classes.append(directoryID)

    # now, save the training and data
    train_set = (train_images, train_classes)
    test_set = (test_images, test_classes)
    data = (train_set, test_set)
    file_name = 'D:\\_Dataset\\GTSRB\\gtsrb_serialized_%s_%d.pkl' % (type_char, img_dim)

    # this may cause memory problems
    # pickle.dump(data, open(file_name, 'wb'))

    p = pickle._Pickler(open(file_name, "wb"))
    p.fast = True
    p.dump(data)

    print("Finish Preparing Data")


def organize_gtsr(img_dim, superclass_type):
    """
    Read the reduced dataset (it contains only 10 classes out of 43)
    then split the training to training and validation, the save it on disk

    :return:
    """

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        classes_ids = CNN.consts.ClassesIDs.PROHIB_CLASSES
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        classes_ids = CNN.consts.ClassesIDs.WARNING_CLASSES
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        classes_ids = CNN.consts.ClassesIDs.MANDATORY_CLASSES
        type_char = 'm'
    elif superclass_type == CNN.enums.SuperclassType._04_Other:
        classes_ids = CNN.consts.ClassesIDs.OTHER_CLASSES
        type_char = 'o'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    file_name = 'D:\\_Dataset\\GTSRB\\gtsrb_serialized_%s_%d.pkl' % (type_char, img_dim)
    data = pickle.load(open(file_name, 'rb'))

    tr_images = data[0][0]
    tr_classes = data[0][1]

    train_images = []
    train_classes = []
    valid_images = []
    valid_classes = []

    tr_images_reshaped = []
    for i in range(len(classes_ids)):
        tr_images_reshaped.append([])

    for i in range(len(tr_classes)):
        index = classes_ids.index(tr_classes[i])
        tr_images_reshaped[index].append(tr_images[i])

    for i in range(len(classes_ids)):
        class_id = classes_ids[i]
        n = tr_classes.count(class_id)
        nTrain = int(n * 3 / 4)
        nValid = n - nTrain

        # create shuffled indexes to shuffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * class_id).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nValid,), dtype=int) * class_id).tolist()
        valid_images.extend(images)
        valid_classes.extend(classes)

    # shuffle the train and valid dateset
    idx = numpy.arange(start=0, stop=len(train_classes), dtype=int).tolist()
    random.shuffle(idx)
    train_images_shuffled = [train_images[j] for j in idx]
    train_classes_shuffled = [train_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(valid_classes), dtype=int).tolist()
    random.shuffle(idx)
    valid_images_shuffled = [valid_images[j] for j in idx]
    valid_classes_shuffled = [valid_classes[j] for j in idx]

    # change array to numpy
    # For all the images, cast the ndarray from int to float64 then normalize (divide by 255)
    train_images = numpy.asarray(train_images_shuffled, dtype=float)
    train_classes = numpy.asarray(train_classes_shuffled)
    valid_images = numpy.asarray(valid_images_shuffled, dtype=float)
    valid_classes = numpy.asarray(valid_classes_shuffled)
    test_images = numpy.asarray(data[1][0], dtype=float)
    test_classes = numpy.asarray(data[1][1])

    # now, save the training and data
    file_name = 'D:\\_Dataset\\GTSRB\\gtsrb_organized_%s_%d.pkl' % (type_char, img_dim)
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))

    # this line of code can trigger memory warning
    # pickle.dump(data, open(file_name, 'wb'))

    # if memory warning happen, try this
    p = pickle._Pickler(open(file_name, "wb"))
    p.fast = True
    p.dump(data)

    print("Finish Preparing Data")


def map_class_ids(img_dim, superclass_type):
    # because the ids of the data are not successive
    # so we need to map them

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        classes_ids = CNN.consts.ClassesIDs.PROHIB_CLASSES
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        classes_ids = CNN.consts.ClassesIDs.WARNING_CLASSES
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        classes_ids = CNN.consts.ClassesIDs.MANDATORY_CLASSES
        type_char = 'm'
    elif superclass_type == CNN.enums.SuperclassType._04_Other:
        classes_ids = CNN.consts.ClassesIDs.OTHER_CLASSES
        type_char = 'o'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    file_path = 'D:\\_Dataset\\GTSRB\\gtsrb_organized_%s_%d.pkl' % (type_char, img_dim)
    data = pickle.load(open(file_path, 'rb'))

    train_classes = numpy.zeros(shape=(0, 1), dtype=int)
    valid_classes = numpy.zeros(shape=(0, 1), dtype=int)
    test_classes = numpy.zeros(shape=(0, 1), dtype=int)
    all_classes = [train_classes, valid_classes, test_classes]

    for i in range(0, len(data)):
        classes = data[i][1]
        for id in classes_ids:
            classes[classes == id] = id * 100

        for j in range(0, len(classes_ids)):
            id = classes_ids[j] * 100
            classes[classes == id] = j

        all_classes[i] = classes

    data = ((data[0][0], all_classes[0]), (data[1][0], all_classes[1]), (data[2][0], all_classes[2]))
    pickle.dump(data, open(file_path, 'wb'))


def restore_class_ids(mapped_ids, superclass_type):
    """
     because the class ids of the database was mapped (for example from [33, 35, 37] to [0 1 2])
     this function restores the original ids
    :return:
    """

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        classes_ids = CNN.consts.ClassesIDs.PROHIB_CLASSES
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        classes_ids = CNN.consts.ClassesIDs.WARNING_CLASSES
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        classes_ids = CNN.consts.ClassesIDs.MANDATORY_CLASSES
    elif superclass_type == CNN.enums.SuperclassType._04_Other:
        classes_ids = CNN.consts.ClassesIDs.OTHER_CLASSES
    else:
        raise Exception("Sorry, un-recognized super-class type")

    # copy list
    original_ids = mapped_ids.copy()

    # recover the original ids
    for i in range(0, len(classes_ids)):
        original_ids[mapped_ids == i] = classes_ids[i]

    return original_ids


def __reduce_gtsr(img_dim):
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset
    - only read training data of the first 10 classes only
    - read the test data corresponding to these 10 classes
    - split the training to train and valid testsets to have the same structure as MNIST dataset

    :return:
    '''

    file_name = 'D:\\_Dataset\\GTSRB\\gtsrb_serialized_%d.pkl' % (img_dim)
    data = pickle.load(open(file_name, 'rb'))
    tr_set = data[0]
    test_set = data[1]

    train_images = tr_set[0]
    train_classes = tr_set[1]
    test_images = test_set[0]
    test_classes = test_set[1]

    train_images_reduced = []
    train_classes_reduced = []
    test_images_reduced = []
    test_classes_reduced = []

    # reduce training
    for i in range(0, len(train_images)):
        if train_classes[i] > 9:
            break
        else:
            train_images_reduced.append(train_images[i])
            train_classes_reduced.append(train_classes[i])

    # reduce test
    for i in range(0, len(test_images)):
        if test_classes[i] < 10:
            test_images_reduced.append(test_images[i])
            test_classes_reduced.append(test_classes[i])

    print(len(train_images_reduced))
    print(len(train_classes_reduced))
    print(len(test_images_reduced))
    print(len(test_classes_reduced))

    # now, save the training and data
    train_set = (train_images_reduced, train_classes_reduced)
    test_set = (test_images_reduced, test_classes_reduced)
    data = (train_set, test_set)
    pickle.dump(data, open('D:\\_Dataset\\GTSRB\\gtsrb_reduced.pkl', 'wb'))

    print("Finish Preparing Data")


# endregion

# region BelgiumTS

def serialize_belgiumTS():
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset

    :return:
    '''

    from os import listdir
    from os.path import isfile, join

    train_images = []
    train_classes = []
    test_images = []
    test_classes = []

    directoryTrain = "D:\\_Dataset\\BelgiumTS\\Training_Preprocessed_28\\"
    directoryTest = "D:\\_Dataset\\BelgiumTS\\Test_Preprocessed_28\\"

    # get the training data
    for i in range(0, 62):
        subDirectory = directoryTrain + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            train_images.append(fileData)
            train_classes.append(i)

    # get the test data
    for i in range(0, 62):
        subDirectory = directoryTest + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            test_images.append(fileData)
            test_classes.append(i)

    # now, save the training and data
    train_set = (train_images, train_classes)
    test_set = (test_images, test_classes)
    data = (train_set, test_set)

    pickle.dump(data, open('D:\\_Dataset\\BelgiumTS\\BelgiumTS.pkl', 'wb'))

    print("Finish Preparing Data")


def reduce_belgiumTS():
    '''
    Read the preprocessed images (training and test) and save them on the disk
    Save them with the same format and data structure as the MNIST dataset
    - only read training data of the first 10 classes only
    - read the test data corresponding to these 10 classes
    - split the training to train and valid testsets to have the same structure as MNIST dataset

    :return:
    '''

    data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS.pkl', 'rb'))

    tr_set = data[0]
    test_set = data[1]

    train_images = tr_set[0]
    train_classes = tr_set[1]
    test_images = test_set[0]
    test_classes = test_set[1]

    train_images_reduced = []
    train_classes_reduced = []
    test_images_reduced = []
    test_classes_reduced = []

    # reduce training
    for i in range(0, len(train_images)):
        if train_classes[i] > 9:
            break
        else:
            train_images_reduced.append(train_images[i])
            train_classes_reduced.append(train_classes[i])

    # reduce test
    for i in range(0, len(test_images)):
        if test_classes[i] > 9:
            break
        else:
            test_images_reduced.append(test_images[i])
            test_classes_reduced.append(test_classes[i])

    print(len(train_images_reduced))
    print(len(train_classes_reduced))
    print(len(test_images_reduced))
    print(len(test_classes_reduced))

    # now, save the training and data
    train_set = (train_images_reduced, train_classes_reduced)
    test_set = (test_images_reduced, test_classes_reduced)
    data = (train_set, test_set)
    pickle.dump(data, open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_reduced.pkl', 'wb'))

    print("Finish Preparing Data")


def organize_belgiumTS():
    """
    Read the reducted dataset (it contains only 10 classes out of 43)
    then split the training to training and validation, the save it on disk

    :return:
    """

    data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_reduced.pkl', 'rb'))

    tr_set = data[0]

    tr_images = tr_set[0]
    tr_classes = tr_set[1]

    train_images = []
    train_classes = []
    valid_images = []
    valid_classes = []
    test_images = data[1][0]
    test_classes = data[1][1]

    del data

    tr_images_reshaped = []
    for i in range(10):
        tr_images_reshaped.append([])

    for i in range(len(tr_images)):
        tr_images_reshaped[tr_classes[i]].append(tr_images[i])

    for i in range(10):
        n = tr_classes.count(i)
        nTrain = int(n * 3 / 4)
        nValid = n - nTrain

        # create shuffled indexes to shuffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * i).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nValid,), dtype=int) * i).tolist()
        valid_images.extend(images)
        valid_classes.extend(classes)

    # shuffle the train and valid dateset
    idx = numpy.arange(start=0, stop=len(train_classes), dtype=int).tolist()
    random.shuffle(idx)
    train_images_shuffled = [train_images[j] for j in idx]
    train_classes_shuffled = [train_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(valid_classes), dtype=int).tolist()
    random.shuffle(idx)
    valid_images_shuffled = [valid_images[j] for j in idx]
    valid_classes_shuffled = [valid_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(test_classes), dtype=int).tolist()
    random.shuffle(idx)
    test_images_shuffled = [test_images[j] for j in idx]
    test_classes_shuffled = [test_classes[j] for j in idx]

    # change array to numpy
    train_images = numpy.asarray(train_images_shuffled)
    train_classes = numpy.asarray(train_classes_shuffled)
    valid_images = numpy.asarray(valid_images_shuffled)
    valid_classes = numpy.asarray(valid_classes_shuffled)
    test_images = numpy.asarray(test_images_shuffled)
    test_classes = numpy.asarray(test_classes_shuffled)

    # For all the images, cast the ndarray from int to float64 then normalize (divide by 255)
    train_images = train_images.astype(float) / 255.0
    valid_images = valid_images.astype(float) / 255.0
    test_images = test_images.astype(float) / 255.0

    # now, save the training and data
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))
    pickle.dump(data, open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_normalized.pkl', 'wb'))

    print("Finish Preparing Data")


def serialize_belgiumTS_non_gtsrb():
    # serialize images in Belgium Traffic Sign Dataset
    # that does not exist in GTSRB
    directoryTrain = "D:\\_Dataset\\BelgiumTS\\Training_Preprocessed_28\\"
    directoryTest = "D:\\_Dataset\\BelgiumTS\\Test_Preprocessed_28\\"

    images = []
    classes = []
    img_dim = 28

    # prohibitory, warning, mandatory
    classes_ids = [[20, 29, 30], [3, 4, 5, 9, 12, 14, 15, 18], [35]]
    # classes_ids = [[25, 28, 31], [11, 13, 17], [37]]
    for i in range(0, len(classes_ids)):
        for folder_id in classes_ids[i]:
            subDirectory = directoryTest + "{0:05d}\\".format(folder_id)
            onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
            for file in onlyfiles:
                fileName = join(subDirectory, file)
                img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
                images.append(img)
                classes.append(i)

    images = numpy.asarray(images, dtype=float) / 255.0
    images.reshape((images.shape[0], img_dim * img_dim))
    classes = numpy.asarray(classes, dtype=int)
    data = (images, classes)
    file_path = "D:\\_Dataset\BelgiumTS\\BelgiumTS_non_GTSRB_28.pkl"
    with open(file_path, "wb") as data_file:
        pickle.dump(data, data_file)


# endregion

# region SuperClass


def serialize_SuperClass_old():
    '''
    Read the preprocessed images (training and test), then split the training to training and validation
    then save them on the disk, Save them with the same format and data structure as the MNIST dataset

    :return:
    '''

    tr_images = []
    tr_classes = []
    test_images = []
    test_classes = []

    directoryTrain = "D:\\_Dataset\\SuperClass\\Training_Preprocessed_Revised\\"
    directoryTest = "D:\\_Dataset\\SuperClass\\Test_Preprocessed_Revised\\"

    nClasses = 3

    # get the training data
    for i in range(0, nClasses):
        subDirectory = directoryTrain + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            tr_images.append(fileData)
            tr_classes.append(i)

    # get the test data
    for i in range(0, nClasses):
        subDirectory = directoryTest + "{0:05d}\\".format(i)
        onlyfiles = [f for f in listdir(subDirectory) if isfile(join(subDirectory, f))]
        for file in onlyfiles:
            fileName = join(subDirectory, file)
            fileData = numpy.asarray(PIL.Image.open(fileName).getdata())
            test_images.append(fileData)
            test_classes.append(i)

    # split the tr to train and valid
    # normalize the images and save as double
    train_images = []
    train_classes = []
    valid_images = []
    valid_classes = []

    tr_images_reshaped = []
    for i in range(10):
        tr_images_reshaped.append([])

    for i in range(len(tr_images)):
        tr_images_reshaped[tr_classes[i]].append(tr_images[i])

    for i in range(nClasses):
        n = tr_classes.count(i)
        nTrain = int(n * 4 / 5)
        nValid = n - nTrain

        # create shuffled indexes to shuffle the train images and classes
        idx = numpy.arange(start=0, stop=n, dtype=int).tolist()
        random.shuffle(idx)

        # take the first nTrain items as train_set
        idxRange = numpy.arange(start=0, stop=nTrain, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nTrain,), dtype=int) * i).tolist()
        train_images.extend(images)
        train_classes.extend(classes)

        # take the next nValid items as validation_set
        idxRange = numpy.arange(start=nTrain, stop=n, dtype=int).tolist()
        images = [tr_images_reshaped[i][j] for j in idxRange]
        classes = (numpy.ones(shape=(nValid,), dtype=int) * i).tolist()
        valid_images.extend(images)
        valid_classes.extend(classes)

    # shuffle the train and valid dateset
    idx = numpy.arange(start=0, stop=len(train_classes), dtype=int).tolist()
    random.shuffle(idx)
    train_images_shuffled = [train_images[j] for j in idx]
    train_classes_shuffled = [train_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(valid_classes), dtype=int).tolist()
    random.shuffle(idx)
    valid_images_shuffled = [valid_images[j] for j in idx]
    valid_classes_shuffled = [valid_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(test_classes), dtype=int).tolist()
    random.shuffle(idx)
    test_images_shuffled = [test_images[j] for j in idx]
    test_classes_shuffled = [test_classes[j] for j in idx]

    # change array to numpy
    train_images = numpy.asarray(train_images_shuffled)
    train_classes = numpy.asarray(train_classes_shuffled)
    valid_images = numpy.asarray(valid_images_shuffled)
    valid_classes = numpy.asarray(valid_classes_shuffled)
    test_images = numpy.asarray(test_images_shuffled)
    test_classes = numpy.asarray(test_classes_shuffled)

    # For all the images, cast the ndarray from int to float64 then normalize (divide by 255)
    train_images = train_images.astype(float) / 255.0
    valid_images = valid_images.astype(float) / 255.0
    test_images = test_images.astype(float) / 255.0

    # now, save the training and data
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))
    pickle.dump(data, open('D:\\_Dataset\\SuperClass\\SuperClass_normalized.pkl', 'wb'))

    print("Finish Preparing Data")


def serialize_superclass(img_dim):
    '''
    Collect the images of GTSRB into superclasses
    :return:
    '''

    directory_train = "D:\\_Dataset\\GTSRB\Final_Training_Preprocessed\\"
    directory_test = "D:\\_Dataset\\GTSRB\\Final_Test_Preprocessed\\"

    n_classes = 4
    superclasses_ids = [CNN.consts.ClassesIDs.PROHIB_CLASSES,
                        CNN.consts.ClassesIDs.WARNING_CLASSES,
                        CNN.consts.ClassesIDs.MANDATORY_CLASSES,
                        CNN.consts.ClassesIDs.OTHER_CLASSES]

    tr_images = []
    tr_classes = []
    test_images = []
    test_classes = []
    test_classes_old = []

    # read the csv of the test images to
    # get the ground truth of the test data
    # get the ground truth as superclass id not class id
    csvFileName = "D:\\_Dataset\\GTSRB\\Final_Test_PNG\\GT-final_test.annotated.csv"
    test_images_names = []
    with open(csvFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if row[7] != "ClassId":
                test_images_names.append(row[0][:-4])
                class_id = int(row[7])
                test_classes_old.append(class_id)
                # for i in range(0, len(superclasses_ids)):
                #     classes_ids = superclasses_ids[i]
                #     if class_id in classes_ids:
                #         test_classes.append(i)
                #         break
                if class_id in superclasses_ids[0]:
                    test_classes.append(0)
                elif class_id in superclasses_ids[1]:
                    test_classes.append(1)
                elif class_id in superclasses_ids[2]:
                    test_classes.append(2)
                elif class_id in superclasses_ids[3]:
                    test_classes.append(3)
                else:
                    raise Exception("Unknown class")

    # loop on images of each super class, then resize and append them
    # for example loop on the images in the folders [0, 1, 2, 3, .....]
    # and add them to the prohibitory superclass
    for i in range(0, len(superclasses_ids)):
        classes_ids = superclasses_ids[i]
        for ids in classes_ids:
            sub_directory = directory_train + "{0:05d}\\".format(ids)
            onlyfiles = [f for f in listdir(sub_directory) if isfile(join(sub_directory, f))]
            # we don't want to take all the images in the class, we want only small sample
            n_files = len(onlyfiles)
            n_samples = 150
            if n_files < n_samples:
                n_samples = n_files
            idx = numpy.arange(start=0, stop=n_files, dtype=int).tolist()
            random.shuffle(idx)
            for f_index in idx[0:n_samples]:
                file = onlyfiles[f_index]
                fileName = join(sub_directory, file)
                img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (img_dim, img_dim))
                tr_images.append(img)
                tr_classes.append(i)

    # loop on all the images of the test, resize and append
    onlyfiles = [f for f in listdir(directory_test) if isfile(join(directory_test, f))]
    for file in onlyfiles:
        fileName = join(directory_test, file)
        img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (img_dim, img_dim))
        test_images.append(img)

    # shuffle the train and valid dateset
    idx = numpy.arange(start=0, stop=len(tr_classes), dtype=int).tolist()
    random.shuffle(idx)
    tr_images_shuffled = [tr_images[j] for j in idx]
    tr_classes_shuffled = [tr_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(test_classes), dtype=int).tolist()
    random.shuffle(idx)
    test_images_shuffled = [test_images[j] for j in idx]
    test_classes_shuffled = [test_classes[j] for j in idx]

    # split tr to train and test
    n = len(tr_classes)
    nTrain = int(n * 4 / 5)
    train_images_shuffled = tr_images_shuffled[0:nTrain]
    train_classes_shuffled = tr_classes_shuffled[0:nTrain]
    valid_images_shuffled = tr_images_shuffled[nTrain:n]
    valid_classes_shuffled = tr_classes_shuffled[nTrain:n]

    # change array to numpy
    train_images = numpy.asarray(train_images_shuffled)
    train_classes = numpy.asarray(train_classes_shuffled)
    valid_images = numpy.asarray(valid_images_shuffled)
    valid_classes = numpy.asarray(valid_classes_shuffled)
    test_images = numpy.asarray(test_images_shuffled)
    test_classes = numpy.asarray(test_classes_shuffled)

    train_images = train_images.reshape((train_images.shape[0], img_dim * img_dim))
    valid_images = valid_images.reshape((valid_images.shape[0], img_dim * img_dim))
    test_images = test_images.reshape((test_images.shape[0], img_dim * img_dim))

    # For all the images, cast the ndarray from int to float64 then normalize (divide by 255)
    train_images = train_images.astype(float) / 255.0
    valid_images = valid_images.astype(float) / 255.0
    test_images = test_images.astype(float) / 255.0

    # now, save the training and data
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))
    data_path = "D:\\_Dataset\\SuperClass\\SuperClass_organized_%d.pkl" % (img_dim)
    pickle.dump(data, open(data_path, 'wb'))

    print("Finish Preparing Data")


# endregion

# region GTSD


def serialize_gtsdb(img_dim, superclass_type, add_true_negative=True, pre_processing=True):
    """
    read the german traffic sign detection database
    for each image, create multiple scales
    for each scale, create regions/batches around the ground truth boundary
    all these created regions must comprise completely the ground truth
    re-calculate the x,y of the ground truth, instead of the whole image
    is the frame_of_reference, the region itself is the frame_of_reference
    :param img_dim:
    :param add_true_negative:
    :return:
    """

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        type_char = 'm'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    # get the ground truth of the test data
    csv_data = []
    csvFileName = "D:\\_Dataset\\GTSDB\\Ground_Truth\\gt.txt"
    with open(csvFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            col_data = []
            for col in row:
                if len(col_data) == 0:
                    col_data.append(int(col[:-4]))
                else:
                    col_data.append(int(col))

            csv_data.append(col_data)

    # stride represents how dense to sample regions around the ground truth traffic signs
    # also up_scaling factor affects the sampling
    # initial dimension defines what is the biggest traffic sign to recognise
    # actually stride should be dynamic, i.e. smaller strides for smaller window size and vice versa
    # stride_factor = 2 gives more sampling than stride_factor = 1
    # don't use value smaller than 1 for stride factor
    up_scale_factor = 1.2
    up_scale_increment = 10
    biggest_area_factor = 4 ** 2
    min_region_dim = img_dim * 2 / 3
    img_width = 1630
    img_height = 800
    stride_factor = 10
    regions = numpy.zeros(shape=(0, img_dim * img_dim), dtype=float)
    relative_boundaries = numpy.zeros(shape=(0, 4), dtype=int)

    directory = "D:\\_Dataset\\GTSDB\\Training_PNG\\"
    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    for file in files:
        file_id = int(file[:-4])

        # print for debugging
        print("... file: %d, regions: %d" % (file_id, regions.shape[0]))

        file_path = join(directory, file)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # loop on each ground truth (i.e traffic sign)
        # for now, consider only traffic signs of the given superclass
        # get the ground truth of 'warning' signs
        boundaries = __gtsr_get_boundaries(csv_data, file_id, superclass_type)
        for boundary in boundaries:
            # the biggest traffic sign to recognize is 400*400 in a 1360*800 image
            # that means, we'll start with a window with initial size of the ground_truth
            # for each ground_truth boundary, extract regions in such that:
            # 1. each region fully covers the boundary
            # 2. the boundary must not be bigger than nth (biggest_area_factor) of the region
            # we will recognize smaller ground truth because we up_scale the window every step
            # also, we will recognize bigger ground_truth because the first region taken for
            # a ground_truth is almost it's size
            # boundary is x1, y1, x2, y2 => (x1,y1) top left, (x2, y2) bottom right
            # don't forget that stride of sliding the window is dynamic
            x1 = boundary[0]
            y1 = boundary[1]
            x2 = boundary[2]
            y2 = boundary[3]
            boundary_width = x2 - x1
            boundary_height = y2 - y1
            boundary_max_dim = max(boundary_width, boundary_height)
            window_dim = boundary_max_dim
            boundary_area = boundary_width * boundary_height
            while (window_dim ** 2 / boundary_area) <= biggest_area_factor \
                    and (boundary_max_dim * img_dim / window_dim) >= min_region_dim \
                    and window_dim <= img_height and window_dim <= img_width:

                # for the current scale of the window, extract regions
                # please don't change this sentence, it's magic
                # it means that the stride is boundary size independent (boundary_max_dim / img_dim)
                # also, it means the stride gets bigger when the ratio of boundary to window gets smaller (window_dim / boundary_max_dim)
                # stride = int(stride_factor * (boundary_max_dim / img_dim) * (window_dim / boundary_max_dim))
                stride = int(stride_factor * window_dim / img_dim)
                y_range = numpy.arange(start=y2 - window_dim, stop=y1 + 1, step=stride, dtype=int).tolist()
                x_range = numpy.arange(start=x2 - window_dim, stop=x1 + 1, step=stride, dtype=int).tolist()

                # for debugging
                # print("stride: %d, boundary: %d, window: %d" % (stride, boundary_max_dim, window_dim))

                # if last x in x_range don't make the sliding window reach the end of the region
                # then add one more x   to the x_range to satisfy this
                x_r_len = len(x_range)
                y_r_len = len(y_range)
                if int((x1 - x_range[x_r_len - 1]) * boundary_width / window_dim) > 1:
                    x_range.append(x1)
                if int((y1 - y_range[y_r_len - 1]) * boundary_height / window_dim) > 1:
                    y_range.append(y1)

                # the factor used to rescale the region before saving it to the region array
                # note that we want to rescale to 28*28 to be compatible with our CNN recognition network
                r_factor = window_dim / img_dim
                for y in y_range:
                    for x in x_range:

                        # make sure that the window in the current position (x, y) is within the image itself
                        if x < 0 or y < 0 or (x + window_dim) > img_width or (y + window_dim) > img_height:
                            continue

                        # if using the extra-stride in the range makes the region get out of the window
                        # then rebound the window so we can take the last region before exiting the loop

                        # - add region to the region list
                        # - adjust the position of the ground_truth to be relative to the window
                        #   not relative to the image (i.e relative frame of reference)
                        # - don't forget to re_scale the extracted/sampled region to be 28*28
                        #   hence, multiply the relative position with this scaling accordingly
                        # - also, the image needs to be preprocessed so it can be ready for the CNN
                        # - reshape the region to 1-D vector to align with the structure of MNIST database

                        relative_boundary = (numpy.asarray([x1 - x, y1 - y, x2 - x, y2 - y]) / r_factor).astype(int)
                        relative_boundaries = numpy.vstack([relative_boundaries, relative_boundary])

                        region = numpy.copy(img[y:y + window_dim, x:x + window_dim])
                        region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))

                        # pre-process the region if needed
                        if pre_processing:
                            region = skimage.exposure.equalize_hist(region)
                            # region = skimage.exposure.rescale_intensity(region, in_range=(0.1, 0.8))

                        # append the region
                        region = region.reshape((img_dim * img_dim,))
                        regions = numpy.vstack([regions, region])

                        # # save region for experimenting/debugging
                        # filePathWrite = "D:\\_Dataset\\GTSDB\\Training_Regions\\" + file[:-4] + "_" + "{0:05d}.png".format(len(regions))
                        # cv2.imwrite(filePathWrite, region.reshape((img_dim, img_dim)) * 255)

                if add_true_negative:
                    # add some true negatives to increase variance of the machine
                    # add only n images per scale per image, n = 2
                    regions_negatives = __sample_true_negatives(img, window_dim, img_dim, boundaries, 4, pre_processing)
                    regions = numpy.vstack([regions, regions_negatives])
                    relative_boundaries = numpy.vstack([relative_boundaries, numpy.zeros(shape=(regions_negatives.shape[0], 4), dtype=int)])

                # also add relative boundaries for them as the ground truth
                # which should only be zeros

                # # # saving images for experimenting/debugging
                # ss_count = 0
                # for s in regions_negatives:
                #     ss_count += 1
                #     filePathWrite = "D:\\_Dataset\\GTSDB\\Training_Regions\\" + file[:-4] + "_" + "{0:03d}".format(len(regions)) + "{0:02d}.png".format(ss_count)
                #     cv2.imwrite(filePathWrite, s.reshape((img_dim, img_dim)) * 255)

                # now we up_scale the window
                # instead of scaling up by factor, scale up by fixed increment
                # window_dim = int(window_dim * up_scale_factor)
                # please don't change this expression, it's magic
                window_dim += int(up_scale_increment * boundary_max_dim / min_region_dim)

            dummy_variable_end_file_loop = 10

    # dump the regions into a pickle file
    data = (regions, relative_boundaries)
    file_name = 'D:\\_Dataset\\GTSDB\\gtsdb_serialized_%s_%d.pkl' % (type_char, img_dim)
    pickle.dump(data, open(file_name, 'wb'))

    print("Total number of regions: %d" % (regions.shape[0]))
    print("Finish sampling regions for detector")


def organize_gtsdb(img_dim, superclass_type):
    """
    Split the given training to training and valid
    Also, shuffle the training and valid sets
    :param img_dim:
    :return:
    """

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        type_char = 'm'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    file_name = 'D:\\_Dataset\\GTSDB\\gtsdb_serialized_%s_%d.pkl' % (type_char, img_dim)
    data = pickle.load(open(file_name, 'rb'))
    regions = data[0]
    boundaries = data[1]
    del data

    n = len(boundaries)
    nTrain = int(n * 2 / 4)
    nValid = int(n * 1 / 4)

    # may be try to shuffle the data before dividing
    train_images = regions[0:nTrain]
    train_classes = boundaries[0:nTrain]
    valid_images = regions[nTrain:nTrain + nValid]
    valid_classes = boundaries[nTrain:nTrain + nValid]
    test_images = regions[nTrain + nValid:n]
    test_classes = boundaries[nTrain + nValid:n]

    # shuffle the train, valid and test dateset
    idx = numpy.arange(start=0, stop=len(train_classes), dtype=int).tolist()
    random.shuffle(idx)
    train_images_shuffled = [train_images[j] for j in idx]
    train_classes_shuffled = [train_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(valid_classes), dtype=int).tolist()
    random.shuffle(idx)
    valid_images_shuffled = [valid_images[j] for j in idx]
    valid_classes_shuffled = [valid_classes[j] for j in idx]

    idx = numpy.arange(start=0, stop=len(test_classes), dtype=int).tolist()
    random.shuffle(idx)
    test_images_shuffled = [test_images[j] for j in idx]
    test_classes_shuffled = [test_classes[j] for j in idx]

    # change array to numpy
    train_images = numpy.asarray(train_images_shuffled)
    train_classes = numpy.asarray(train_classes_shuffled)
    valid_images = numpy.asarray(valid_images_shuffled)
    valid_classes = numpy.asarray(valid_classes_shuffled)
    test_images = numpy.asarray(test_images_shuffled)
    test_classes = numpy.asarray(test_classes_shuffled)

    # now, save the training and data
    data = ((train_images, train_classes), (valid_images, valid_classes), (test_images, test_classes))
    file_name = 'D:\\_Dataset\\GTSDB\\gtsdb_organized_%s_%d.pkl' % (type_char, img_dim)
    pickle.dump(data, open(file_name, 'wb'))

    print("Finish Preparing Data")


def convolve_gtsdb(recognition_model_path, superclass_type):
    # do all the cov+pool computation using theano
    # while train the regressor of the detector using nolearn and lasagne
    # don't forget to operate on batches becuase:
    # 1. you can't convolve all the training image in once shot
    # 2. to train better regressor

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        type_char = 'm'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    # load model and read it's parameters
    # the same weights of the convolutional layers will be used
    # in training the detector
    loaded_objects = load_model(recognition_model_path, CNN.enums.ModelType._02_conv3_mlp2)
    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]

    # load the data and concatenate all the images together
    print('... loading data')
    dataset_path = "D:\\_Dataset\\GTSDB\\gtsdb_organized_%s_80.pkl" % (type_char)
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # concatenate validation and training sets
    n_train = dataset[0][1].shape[0]
    n_valid = dataset[1][1].shape[0]
    n_test = dataset[2][1].shape[0]
    n_total = n_train + n_valid + n_test
    images = numpy.vstack([dataset[0][0], dataset[1][0], dataset[2][0]])
    n_batches = 10
    batch_size = int(n_total / n_batches)

    # use the weights of the filters of the given recognition_model_path
    # to filter (convolving+downsample) the given input images
    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    layer0_input = theano.tensor.tensor4('input')
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

    ##############################
    # Train The Regression Model #
    ##############################
    # Finally, launch the training loop.
    print("... convolving the data")
    start_time = time.clock()
    all_filters = numpy.zeros(shape=(0, layer3_input_shape[1]), dtype="float32")
    batch_count = 0
    for start_idx in range(0, images.shape[0] - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        images_reshaped = images[excerpt].reshape(-1, 1, layer0_img_dim, layer0_img_dim)
        filters = conv_fn(images_reshaped)
        filters = filters.reshape(layer3_input_shape).astype("float32")
        all_filters = numpy.vstack([all_filters, filters])
        batch_count += 1
        print("... finish convolving batch %d / %d" % (batch_count, n_batches))

    dataset_path = "D:\\_Dataset\\GTSDB\\gtsdb_convolved_%s_80.pkl" % (type_char)
    new_n_test = n_test - (n_total - n_batches * batch_size)
    train_x = all_filters[0:n_train]
    valid_x = all_filters[n_train:n_train + n_valid]
    test_x = all_filters[n_train + n_valid:n_total]
    train_y = dataset[0][1][0:n_train]
    valid_y = dataset[1][1][0:n_valid]
    test_y = dataset[2][1][0: new_n_test]
    dataset = ((train_x, train_y), (valid_x, valid_y), (test_x, test_y))
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    print("... finish convolving ans saving the images, total time consumed: %f" % (duration))


def change_target_to_binary(img_dim, superclass_type, is_convolved_dataset):
    if is_convolved_dataset:
        name = "convolved"
    else:
        name = "organized"

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        type_char = 'm'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    file_name = 'D:\\_Dataset\\GTSDB\\gtsdb_%s_%s_%d.pkl' % (name, type_char, img_dim)
    data = pickle.load(open(file_name, 'rb'))
    y_train = data[0][1]
    y_valid = data[1][1]
    y_test = data[2][1]

    y_train_new = []
    y_valid_new = []
    y_test_new = []

    for y in y_train:
        if numpy.count_nonzero(y) == 0:
            y_train_new.append(0)
        else:
            y_train_new.append(1)
    y_train_new = numpy.asarray(y_train_new, dtype=int)

    for y in y_valid:
        if numpy.count_nonzero(y) == 0:
            y_valid_new.append(0)
        else:
            y_valid_new.append(1)
    y_valid_new = numpy.asarray(y_valid_new, dtype=int)

    for y in y_test:
        if numpy.count_nonzero(y) == 0:
            y_test_new.append(0)
        else:
            y_test_new.append(1)
    y_test_new = numpy.asarray(y_test_new, dtype=int)

    data = ((data[0][0], y_train_new), (data[1][0], y_valid_new), (data[2][0], y_test_new))
    file_name = 'D:\\_Dataset\\GTSDB\\gtsdb_%s_%s_%d_binary.pkl' % (name, type_char, img_dim)
    pickle.dump(data, open(file_name, 'wb'))

    print("Finish converting target to binary")


def __sample_true_negatives(img, window_dim, resize_dim, boundaries, count, pre_processing):
    img_h = img.shape[0]
    img_w = img.shape[1]
    n_boundary = len(boundaries)

    regions = numpy.zeros(shape=(0, resize_dim * resize_dim), dtype=float)
    while regions.shape[0] < count:
        # create a region in a random/stochastic way
        # the region be with the same window_dim
        # and does not intersect with any of the boundaries
        y1 = random.randint(0, img_h - window_dim)
        y2 = y1 + window_dim
        x1 = random.randint(0, img_w - window_dim)
        x2 = x1 + window_dim

        # now, check if any of the boundaries intersect with the region
        n_conditions = 0
        for boundary in boundaries:
            b_x1 = boundary[0]
            b_y1 = boundary[1]
            b_x2 = boundary[2]
            b_y2 = boundary[3]
            if x2 <= b_x1 or x1 >= b_x2 or y2 <= b_y1 or y1 >= b_y2:
                n_conditions += 1
            else:
                break
        if n_conditions == n_boundary:
            region = numpy.copy(img[y1:y2, x1:x2])
            region = skimage.transform.resize(region, output_shape=(resize_dim, resize_dim))

            # pre-process the region if needed
            if pre_processing:
                region = skimage.exposure.equalize_hist(region)
                # region = skimage.exposure.rescale_intensity(region, in_range=(0.1, 0.8))

            region = region.reshape((resize_dim * resize_dim,))
            regions = numpy.vstack([regions, region])

    return regions


def __gtsr_get_boundaries(gt_data, image_id, superclass_type=CNN.enums.SuperclassType._00_All):
    '''
    Get the list of ground truth (boundary of a traffic sign) in the image with the given id
    If superclass is provided, then get the ground truth for only this superclass
    Else, get them all
    :param image_id:
    :param superclass_type:
    :return:
    '''

    prohib_classes = CNN.consts.ClassesIDs.PROHIB_CLASSES
    warning_classes = CNN.consts.ClassesIDs.WARNING_CLASSES
    mandatory_classes = CNN.consts.ClassesIDs.MANDATORY_CLASSES
    superclasses = [prohib_classes, warning_classes, mandatory_classes]
    type_id = superclass_type.value - 1

    if type_id == -1:
        items = [f[1:5] for f in gt_data if f[0] == image_id]
    else:
        items = [f[1:5] for f in gt_data if f[0] == image_id and f[5] in superclasses[type_id]]

    return items


# endregion

# region Check Database

def check_database_1():
    data_1 = pickle.load(open('D:\\_Dataset\\GTSRB\\gtsrb_shuffled.pkl', 'rb'))
    data_2 = pickle.load(open("D:\\_Dataset\\mnist.pkl", 'rb'))

    img_1 = data_1[0][0]
    img_2 = data_2[0][0]

    del data_1
    del data_2

    x = 10


def check_database_2():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (German Traffic Sign Recognition GTSR Dataset)
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_normalized_28.pkl', 'rb'))
    valid_imgs = data[1][0]
    del data

    # get first column of the tuple (which represents the image, while second one represents the image class)
    # then get the first image and show it
    photo = valid_imgs[0]

    photoReshaped = photo.reshape((28, 28))
    matplotlib.pyplot.imshow(photoReshaped, cmap=matplotlib.cm.Greys_r)

    # aPhoto = PIL.Image.open(".\data\\test-image.png")
    aPhoto = PIL.Image.open(".\data\\test_00014.ppm")
    data0 = aPhoto.getdata()
    data1 = list(data0)
    data2 = numpy.asarray(data0)
    data3 = numpy.asarray(aPhoto)
    matplotlib.pyplot.imshow(data3)
    hiThere = 10


def check_database_3():
    """
    Loop on random sample of the images and see if their values are correct or not
    :return:
    """

    import matplotlib.pyplot as plt
    import math

    # data = pickle.load(open('D:\\_Dataset\\mnist.pkl', 'rb'))
    # data = pickle.load(open('D:\\_Dataset\\BelgiumTS\\BelgiumTS_normalized_28.pkl', 'rb'))
    # data = pickle.load(open('D:\\_Dataset\\SuperClass\\SuperClass_normalized.pkl', 'rb'))
    data = pickle.load(open('D:\\_Dataset\\GTSRB\\gtsrb_organized_80.pkl', 'rb'))

    images = data[0][0]
    classes = data[0][1]
    del data

    # get first column of the tuple (which represents the image, while second one represents the image class)
    # then get the first image and show it
    idx = numpy.arange(start=0, stop=(len(classes)), step=len(classes) / 12, dtype=int).tolist()
    print(len(classes))

    plt.figure(1)
    plt.ion()
    plt.gray()
    plt.axis('off')

    for i in idx:
        photo = images[i]
        img_dim = photo.shape[0]
        img_dim = math.sqrt(img_dim)
        photo_reshaped = photo.reshape((img_dim, img_dim))
        c = classes[i]
        print(c)
        plt.imshow(photo_reshaped)
        plt.show()
        x = 10


def check_database_4():
    import math

    data = pickle.load(open('D:\\_Dataset\\GTSRB\\gtsrb_organized_m_28.pkl', 'rb'))

    for j in [0, 1, 2]:
        images = data[j][0]
        classes = data[j][1]

        print(images.shape)
        print(classes.shape)
        print("min, max: %f, %f" % (numpy.max(images), numpy.min(images)))
        print("min, max: %f, %f" % (numpy.max(classes), numpy.min(classes)))

        # get first column of the tuple (which represents the image, while second one represents the image class)
        # then get the first image and show it
        idx = numpy.arange(start=0, stop=(len(classes)), step=len(classes) / 30, dtype=int).tolist()
        for i in idx:
            photo = images[i]
            img_dim = photo.shape[0]
            img_dim = math.sqrt(img_dim)
            photo_reshaped = photo.reshape((img_dim, img_dim)) * 255
            c = classes[i]
            cv2.imwrite("D:\\_Dataset\GTSRB\\_checking\\%d_%d_%d.png" % (c, j, i), photo_reshaped)


def check_database_5():
    import math

    data_path = "D:\\_Dataset\BelgiumTS\\BelgiumTS_non_GTSRB_28.pkl"
    data = pickle.load(open(data_path, 'rb'))

    images = data[0]
    classes = data[1]
    img_dim = 28

    print(images.shape)
    print(classes.shape)
    print("min, max: %f, %f" % (numpy.max(images), numpy.min(images)))
    print("min, max: %f, %f" % (numpy.max(classes), numpy.min(classes)))

    # get first column of the tuple (which represents the image, while second one represents the image class)
    # then get the first image and show it
    idx = numpy.arange(start=0, stop=(len(classes)), step=len(classes) / 30, dtype=int).tolist()
    for i in idx:
        photo = images[i]
        photo_reshaped = photo.reshape((img_dim, img_dim)) * 255
        c = classes[i]
        cv2.imwrite("D:\\_Dataset\\BelgiumTS\\_checking\\%d_%d.png" % (c, i), photo_reshaped)


def check_database_detector(img_dim, superclass_type):
    """
    Check if the organized database for the detector is correct or not
    i.e. if the targets correctly describe the image
    i.e. check if ground truth is correct or not
    :param img_dim:
    :param superclass_type:
    :return:
    """

    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory:
        type_char = 'p'
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        type_char = 'w'
    elif superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        type_char = 'm'
    else:
        raise Exception("Sorry, un-recognized super-class type")

    path = 'D:\\_Dataset\\GTSDB\\gtsdb_organized_%s_%d.pkl' % (type_char, img_dim)
    data = pickle.load(open(path, 'rb'))

    for j in [0, 1, 2]:
        images = data[j][0]
        regions = data[j][1]

        # get first column of the tuple (which represents the image, while second one represents the image class)
        # then get the first image and show it
        idx = numpy.arange(start=0, stop=(len(regions)), step=len(regions) / 30, dtype=int).tolist()
        for i in idx:
            img = images[i]
            img_reshaped = img.reshape((img_dim, img_dim)) * 255
            x1, y1, x2, y2 = regions[i]
            region_exist = not (x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0)
            if region_exist:
                cv2.rectangle(img_reshaped, (x1, y1), (x2, y2), 125, 2, -1)
            cv2.imwrite("D:\\_Dataset\GTSDB\\_checking\\%d_%d_%d.png" % (region_exist, j, i), img_reshaped)


def downscale():
    data = pickle.load(open('D:\\_Dataset\\GTSRB\\gtsrb_organized_80.pkl', 'rb'))

    train_images = numpy.zeros(shape=(0, 28, 28), dtype=float)
    valid_images = numpy.zeros(shape=(0, 28, 28), dtype=float)
    test_images = numpy.zeros(shape=(0, 28, 28), dtype=float)

    all_images = [train_images, valid_images, test_images]

    for i in range(0, len(data)):
        print(i)
        imgs = data[i][0]
        imgs = imgs.reshape((imgs.shape[0], 80, 80))
        for img in imgs:
            resized = skimage.transform.resize(img, output_shape=(28, 28))
            resized = resized.reshape((1, 28, 28))
            all_images[i] = numpy.vstack([all_images[i], resized])
        all_images[i] = all_images[i].reshape((all_images[i].shape[0], 28 * 28))

    x = 10
    data = ((all_images[0], data[0][1]), (all_images[1], data[1][1]), (all_images[2], data[2][1]))
    pickle.dump(data, open('D:\\_Dataset\\GTSRB\\gtsrb_organized_28_new.pkl', 'wb'))

# endregion
