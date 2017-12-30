import math
import numpy as np
import scipy.io as spio
import time
from model import model
from predict import predict
#
# %matplotlib inline
np.random.seed(1)

# Loading the dataset
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#
# # Flatten the training and test images
# X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
# X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# # Normalize image vectors
# X_train = X_train_flatten/255.
# X_test = X_test_flatten/255.
# # Convert training and test labels to one hot matrices
# Y_train = convert_to_one_hot(Y_train_orig, 6)
# Y_test = convert_to_one_hot(Y_test_orig, 6)

mat = spio.loadmat('ltv_secondorder_traindataf.mat', squeeze_me=True)

train_x_org=mat["X_train"]
train_x_org=train_x_org.T
train_y=mat["y_train"]
train_y=train_y.reshape(3,train_y.shape[0])
dev_x_org=mat["X_dev"]
dev_x_org=dev_x_org.T
dev_y=mat["y_dev"]
dev_y=dev_y.reshape(3,dev_y.shape[0])
test_x_org=mat["X_test"]
test_x_org=test_x_org.T
test_y=mat["y_test"]
test_y=test_y.reshape(3,test_y.shape[0])
# a = np.array([0,1,2])
# np.tile(a,(3,1))

# print(train_x_org.shape)
# print(train_y.shape)
start_time=time.clock()
np.random.seed(1)

m_train = train_x_org.shape[0]
num_ti = train_x_org.shape[1]
m_test = test_x_org.shape[0]

# eps=1e-12
# mu=np.mean(train_x_org,axis=1,keepdims=True)
# diff=train_x_org-mu
# stddev=np.std(train_x_org,axis=1,keepdims=True)
# train_x=diff/(stddev+eps)

train_x=train_x_org
dev_x=dev_x_org
test_x=test_x_org

print ("number of training examples = " + str(train_x.shape[1]))
print ("number of test examples = " + str(test_x.shape[1]))
print ("X_train shape: " + str(train_x.shape))
print ("Y_train shape: " + str(train_y.shape))
print ("X_test shape: " + str(test_x.shape))
print ("Y_test shape: " + str(test_y.shape))

parameters = model(train_x, train_y, dev_x, dev_y, learning_rate=0.000015,lambd=0.1,num_epochs=1000, minibatch_size=64, print_cost=True)

predict(test_x, test_y, parameters)
end_time=time.clock()
print(end_time-start_time)