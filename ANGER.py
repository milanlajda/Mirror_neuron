
#  Pattern is recognized when it begins with 1 1 1

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [1, 1, 1, 1, 0, 1],
               [1, 1, 1, 1, 0, 1],
               [0, 0, 1, 1, 0, 1],
               [0, 0, 0, 1, 0, 1]])

# output dataset
y = np.array([[1],
               [1],
               [0],
               [0]])

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((6, 4)) - 1
syn1 = 2*np.random.random((4, 4)) - 1
syn2 = 2*np.random.random((4, 1)) - 1


for j in range(60000):

    # forward propagation (ANN guesses the answers)
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))

    # error calculation
    l3_error = y - l3



# in what direction is the target value?
# were we really sure? if so, don't change too much.
    l3_delta = l3_error*nonlin(l3, deriv=True)

# how much did each l1 value contribute to l2 error (according to the weights)?
    l2_error = l3_delta.dot(syn2.T)

# in what direction is the target value?
# were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2, deriv=True)

# how much did each l1 value contribute to l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

# in what direction is the target l2?
# were we really sure? if so don't change much
    l2_delta = l2_error * nonlin(l2, deriv=True)

# in what direction is the target l1?
# were we really sure? if so don't change much
    l1_delta = l1_error*nonlin(l1, deriv=True)


# update weights
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn2 += l2.T.dot(l3_delta)


# test the ANN with nontraining data
# nontrainingdata1 = np.array([ [1, 1, 1, 0, 1, 0],
#                               [1, 1, 1, 0, 0, 1],
#                               [0, 1, 0, 1, 1, 1],
#                               [0, 0, 1, 0, 1, 0]])
# test1s1 = nonlin(np.dot(nontrainingdata1, syn0))
# test1s2 = nonlin(np.dot(test1s1, syn1))
# test1s3 = nonlin(np.dot(test1s2, syn2))
# print("Output of ANGER with nontraining data: ")
# print(test1s3)

# taked the outside data and uses the trained network
def anger(data):

    test1s1 = nonlin(np.dot(data, syn0))
    test1s2 = nonlin(np.dot(test1s1, syn1))
    test1s3 = nonlin(np.dot(test1s2, syn2))

    return test1s3






