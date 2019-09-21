import numpy as np
np.random.seed(1337)

''' Read input files '''
my_data = np.genfromtxt('pkgo_city66_class5_v1.csv', delimiter=',',skip_header=1)

''' The first column to the 199th column is used as input features '''
X_train = my_data[:,0:200]
X_train = X_train.astype('float32')

''' The 200-th column is the answer '''
y_train = my_data[:,200]
Y_train = y_train.astype('int')

''' Shuffle training data '''
from sklearn.utils import shuffle
X_train,Y_train = shuffle(X_train,Y_train,random_state=100)

''' Split training and validation data '''
validation_split = 0.1
split_idx = int(X_train.shape[0]*(1-validation_split))
X_test = X_train[split_idx:]
Y_test = Y_train[split_idx:]
X_train = X_train[:split_idx]
Y_train = Y_train[:split_idx]
print("Training data", X_train.shape)
print("Training label", Y_train.shape)
print("Testing data", X_test.shape)
print("Testing label",Y_test.shape)

''' Batch generator'''
def gen_batches(X, Y, batch_size=16):
    ''' Shuffle training data '''
    X_train,Y_train = shuffle(X, Y, random_state=100)
    while(True):
        for i in range(int(X.shape[0]/batch_size)):
            x_batch = X[(i*batch_size):((i+1)*batch_size)]
            y_batch = Y[(i*batch_size):((i+1)*batch_size)]
            yield x_batch, y_batch
        