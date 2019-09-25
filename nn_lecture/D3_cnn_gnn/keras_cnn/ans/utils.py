from os import listdir
from os.path import isfile, join
import numpy
from keras.utils import np_utils
import sys
height, width, dim = 32, 32, 3
classes = 10

# this function is provided from the official site
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

# from PIL import Image
# def ndarray2image (arr_data, image_fn):
#   img = Image.fromarray(arr_data, 'RGB')
#   img.save(image_fn)

# need pillow package
from scipy.misc import imsave
def ndarray2image (arr_data, image_fn):
    imsave(image_fn, arr_data)

def read_dataset(dataset_path, ouput_type):
    # define the information of images which can be obtained from official website

    ''' read training data '''
    # get the file names which start with "data_batch" (training data)
    train_fns = [fn for fn in listdir(dataset_path) if isfile(join(dataset_path, fn)) & fn.startswith("data_batch")]

    # list sorting
    train_fns.sort()

    # make a glace about the training data
    fn = train_fns[0]
    raw_data = unpickle(dataset_path + fn)

    # type of raw data
    type(raw_data)
    # <type 'dict'>

    # check keys of training data
    raw_data_keys = raw_data.keys()
    # output ['data', 'labels', 'batch_label', 'filenames']

    # check dimensions of ['data']
    raw_data['data'].shape
    # (10000, 3072)

    # concatenate pixel (px) data into one ndarray [img_px_values]
    # concatenate label data into one ndarray [img_lab]
    img_px_values = 0
    img_lab = 0
    for fn in train_fns:
        raw_data = unpickle(dataset_path + fn)
        if fn == train_fns[0]:
            img_px_values = raw_data['data']
            img_lab = raw_data['labels']
        else:
            img_px_values = numpy.vstack((img_px_values, raw_data['data']))
            img_lab = numpy.hstack((img_lab, raw_data['labels']))

    X_train = []
    
    if (ouput_type == "vec"):
        # set X_train as 1d-ndarray (50000,3072)
        X_train = img_px_values
    elif (ouput_type == "img"):
        # set X_train as 3d-ndarray (50000,32,32,3)
        X_train = numpy.asarray([numpy.dstack((r[0:(width*height)].reshape(height,width),
                                               r[(width*height):(2*width*height)].reshape(height,width),
                                               r[(2*width*height):(3*width*height)].reshape(height,width)
                                             )) for r in img_px_values])
    else:
        sys.exit("Error ouput_type")

    Y_train = np_utils.to_categorical(numpy.array(img_lab), classes)

    # check is same or not!
    # lab_eql = numpy.array_equal([(numpy.argmax(r)) for r in Y_train], numpy.array(img_lab))

    # draw one image from the pixel data
    if (ouput_type == "img"):
        ndarray2image(X_train[0],"test_image.png")

    # print the dimension of training data
    print ('X_train shape:', X_train.shape)
    print ('Y_train shape:', Y_train.shape)

    ''' read testing data '''
    # get the file names which start with "test_batch" (testing data)
    test_fns = [fn for fn in listdir(dataset_path) if isfile(join(dataset_path, fn)) & fn.startswith("test_batch")]

    # read testing data
    fn = test_fns[0]
    raw_data = unpickle(dataset_path + fn)
    print ('testing file', dataset_path + fn)

    # type of raw data
    type(raw_data)

    # check keys of testing data
    raw_data_keys = raw_data.keys()
    # ['data', 'labels', 'batch_label', 'filenames']

    img_px_values = raw_data['data']

    # check dimensions of data
    print ("dim(data)", numpy.array(img_px_values).shape)
    # dim(data) (10000, 3072)

    img_lab = raw_data['labels']
    # check dimensions of labels
    print ("dim(labels)",numpy.array(img_lab).shape)
    # dim(data) (10000,)

    if (ouput_type == "vec"):
        X_test = img_px_values
    elif (ouput_type == "img"):
        X_test = numpy.asarray([numpy.dstack((r[0:(width*height)].reshape(height,width),
                                              r[(width*height):(2*width*height)].reshape(height,width),
                                              r[(2*width*height):(3*width*height)].reshape(height,width)
                                            )) for r in img_px_values])
    else:
        sys.exit("Error ouput_type")

    Y_test = np_utils.to_categorical(numpy.array(raw_data['labels']), classes)

    # scale image data to range [0, 1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # print the dimension of training data
    print ('X_test shape:', X_test.shape)
    print ('Y_test shape:', Y_test.shape)
    return X_train, X_test, Y_train, Y_test

import csv
def write_csv(output_fn, fit_log):
    history_fn = output_fn + '.csv'
    with open(history_fn, 'w') as csv_file:
        w = csv.writer(csv_file, lineterminator='\n')
        temp = numpy.array(list(fit_log.history.values()))
        w.writerow(list(fit_log.history.keys()))
        for i in range(temp.shape[1]):
            w.writerow(temp[:,i])
