import CNN.recog
import CNN.detec
import CNN.enums
import CNN.consts
import CNN.utils

import numpy
import pickle
import csv
import time

import lasagne
import lasagne.layers
import nolearn
import nolearn.lasagne


def classify_testset():
    dataset_path = "D:\\_Dataset\GTSRB\\gtsrb_serialized_test_28.pkl"
    sc_model_path = "D:\\_Dataset\\SuperClass\\cnn_model_las_28.pkl"
    prohib_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_p_28.pkl"
    warning_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_w_28.pkl"
    mandat_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_m_28.pkl"
    others_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_las_o_28.pkl"
    result_file_path = "D:\\_Dataset\GTSRB\\_Results\\Result_%d.csv" % int(time.time())
    img_dim = 28
    train_images = []
    train_classes = []

    # load data
    print('... loading data')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    test_truth = dataset[1]
    test_images = dataset[0]
    n_images = test_images.shape[0]
    test_images = test_images.reshape((n_images, 1, img_dim, img_dim))
    final_predictions = numpy.zeros(shape=(n_images,), dtype=int)
    del dataset

    # load models
    print('... loading superclass model')
    with open(sc_model_path, 'rb') as f:
        sc_model = pickle.load(f)

    print('... predicting superclass')
    sc_prediction = sc_model.predict(test_images)
    sc_prediction_list = sc_prediction.tolist()
    del sc_model

    n_models = 4
    models_names = ["prohibitory", "warning", "mandatory", "others"]
    models_pathes = [prohib_model_path, warning_model_path, mandat_model_path, others_model_path]
    superclass_types = [CNN.enums.SuperclassType._01_Prohibitory,
                        CNN.enums.SuperclassType._02_Warning,
                        CNN.enums.SuperclassType._03_Mandatory,
                        CNN.enums.SuperclassType._04_Other]

    # for all the class-specific models
    for i in range(0, n_models):
        # load the model
        print("... loading %s model" % (models_names[i]))
        with open(models_pathes[i], 'rb') as f:
            class_model = pickle.load(f)

        img_idx = numpy.where(sc_prediction == i)[0]
        model_images = test_images[img_idx]

        # get prediction, then delete the model
        class_predictions = class_model.predict(model_images)
        del class_model

        # restore the original class ids
        class_predictions = CNN.utils.restore_class_ids(class_predictions, superclass_types[i])

        # add these predictions in their location of the final list
        final_predictions[img_idx] = class_predictions

    # finally, calcuate the final error
    error = 100 * numpy.sum(numpy.not_equal(final_predictions, test_truth)) / n_images
    print("... error percentage: %f%%" % (error))

    resultdata_file_path = "D:\\_Dataset\GTSRB\\_Results\\Data-Result_%d.pkl" % int(time.time())
    with open(resultdata_file_path, "wb") as result_data_file:
        pickle.dump(final_predictions, result_data_file)

    # now, create submission
    with open(result_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";", quoting=csv.QUOTE_NONE)
        for i in range(0, n_images):
            column1 = "{0:05d}.ppm".format(i)
            column2 = " %d" % final_predictions[i]
            writer.writerow([column1, column2])
