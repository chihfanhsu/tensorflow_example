from os import listdir
from os.path import isfile, join
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

height, width, dim = 32, 32, 3
classes = 10
# this function is provided from the official site
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

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

    Y_train = numpy.array(img_lab)

    # check is same or not!
    # lab_eql = numpy.array_equal([(numpy.argmax(r)) for r in Y_train], numpy.array(img_lab))

    # draw one image from the pixel data
    if (ouput_type == "img"):
        plt.imshow(X_train[0])
        plt.show()
        plt.close()

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

    Y_test = numpy.array(raw_data['labels'])

    # scale image data to range [0, 1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # print the dimension of training data
    print ('X_test shape:', X_test.shape)
    print ('Y_test shape:', Y_test.shape)
    return X_train, X_test, Y_train, Y_test

''' Batch generator'''
def gen_batches(X, Y, batch_size=16):
    ''' Shuffle training data '''
    X_train,Y_train = shuffle(X, Y, random_state=100)
    while(True):
        for i in range(int(X.shape[0]/batch_size)):
            x_batch = X[(i*batch_size):((i+1)*batch_size)]
            y_batch = Y[(i*batch_size):((i+1)*batch_size)]
            yield x_batch, y_batch
