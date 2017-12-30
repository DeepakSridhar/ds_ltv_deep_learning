import numpy as np
import scipy.io as spio
import time
from misc import *
from L_layer_multi_model import L_layer_multi_model
from predict import predict


mat = spio.loadmat('ltv_secondorder_traindatae.mat', squeeze_me=True)

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

### CONSTANTS ###
layers_dims = [1000, 60, 3] #  5-layer model
num_labels=3

parameters = L_layer_multi_model(train_x, train_y, layers_dims, num_labels, num_iterations = 1000, print_cost = True)

pred_train = predict(train_x, train_y, parameters)
pred_dev = predict(dev_x, dev_y, parameters)
pred_test = predict(test_x, test_y, parameters)

end_time=time.clock()
print(end_time-start_time)