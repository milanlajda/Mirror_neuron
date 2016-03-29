import numpy as np


# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [0], [0.12], [0.2], [0.24],
               [0.25], [0.35], [0.4], [0.49],
               [0.5], [0.55], [0.63], [0.74],
               [0.75], [0.83], [0.94], [1] ])

# output dataset
y = np.array([[0, 0, 0, 0,
               1, 1, 1, 1,
               0, 0, 0, 0,
               0, 0, 0, 0]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((1, 1)) - 1

# initialize weights of layer 1 randomly with mean 0
syn1 = 2*np.random.random((1, 1)) - 1

# ################################ layer1
for inter in range(10000):

    # forward propagation 1 (ANN guesses the answers)
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))

    # error calculation
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

# print("Output of l1 after training: ")
# print(l1)

# ################################ layer2
for inter1 in range(10000):

    # forward propagation 2 (ANN guesses the answers)
    l2 = nonlin(np.dot(l1, syn1))

    # error calculation
    l2_error = y - l2

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l2_delta = l2_error * nonlin(l2, True)

    # update weights
    syn1 += np.dot(l1.T, l2_delta)

# print("Output of l2 after training: ")
# print(l2)


# test the ANN with nontraining data
# nontrainingdata1 = np.array([ [0.1], [0.10], [0.15], [0.22] ])
# test1s1 = nonlin(np.dot(nontrainingdata1, syn0))
# test1s2 = nonlin(np.dot(test1s1, syn1))
# print("Output of JOY with JOY data: ")
# print(test1s2)

nontrainingdata2 = np.array([ [0.26], [0.30], [0.41], [0.44] ])
test2s1 = nonlin(np.dot(nontrainingdata2, syn0))
test2s2 = nonlin(np.dot(test2s1, syn1))
print("Output of SADNESS with SADNESS data: ")
print(test2s2)
#
# nontrainingdata3 = np.array([ [0.52], [0.62], [0.69], [0.72] ])
# test3s1 = nonlin(np.dot(nontrainingdata3, syn0))
# test3s2 = nonlin(np.dot(test3s1, syn1))
# print("Output of SADNESS with ANGER data: ")
# print(test3s2)
#
# nontrainingdata4 = np.array([ [0.78], [0.8], [0.92], [0.98] ])
# test4s1 = nonlin(np.dot(nontrainingdata4, syn0))
# test4s2 = nonlin(np.dot(test4s1, syn1))
# print("Output of SADNESS with with FEAR data: ")
# print(test4s2)


def sadness(annoutput):
    if annoutput == 2:
        print("TEST: It's SADNESS!")

    else:
        print("TEST: It's not SADNESS")