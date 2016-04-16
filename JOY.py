
# Pattern is recognized when

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [0, 0, 1, 0, 0, 1],
               [0, 1, 0, 0, 1, 1],
               [1, 0, 1, 1, 0, 1],
               [1, 1, 1, 1, 1, 1]])

# output dataset
y = np.array([[1],
               [1],
               [0],
               [0]])

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((6, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1


# ################################ layer1
for j in range(60000):

    # forward propagation 1 (ANN guesses the answers)
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # error calculation
    l2_error = y - l2

    # if (j% 10000) == 0:
    #     print("Error:" + str(np.mean(np.abs(l2_error))))


# in what direction is the target value?
# were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2, deriv=True)

# how much did each l1 value contribute to l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

# in what direction is the target l1?
# were we really sure? if so don't change much
    l1_delta = l1_error*nonlin(l1, deriv=True)


# update weights
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)

print("Output of l2 in JOY after training: ")
print(l2)

# # ################################ layer2
# for inter1 in range(10000):
#
#     # forward propagation 2 (ANN guesses the answers)
#     l2 = nonlin(np.dot(l1, syn1))
#
#     # error calculation
#     l2_error = y - l2
#
#     # multiply how much we missed by the
#     # slope of the sigmoid at the values in l1
#     l2_delta = l2_error * nonlin(l2, True)
#
#     # update weights
#     syn1 += np.dot(l1.T, l2_delta)

# print("Output of l2 after training: ")
# print(l2)


# test the ANN with nontraining data
# nontrainingdata1 = np.array([ [0.1, 0.09, 0.20], [0.10, 0.9, 0.7], [0.15, 0.247, 0.198], [0.22, 0.23, 0.07] ])
# test1s1 = nonlin(np.dot(nontrainingdata1, syn0))
# # test1s2 = nonlin(np.dot(test1s1, syn1))
# print("Output of JOY with JOY data: ")
# print(test1s1)

# nontrainingdata2 = np.array([ [0.26], [0.30], [0.41], [0.44] ])
# test2s1 = nonlin(np.dot(nontrainingdata2, syn0))
# test2s2 = nonlin(np.dot(test2s1, syn1))
# print("Output of JOY with SADNESS data: ")
# print(test2s2)
#
# nontrainingdata3 = np.array([ [0.52], [0.62], [0.69], [0.72] ])
# test3s1 = nonlin(np.dot(nontrainingdata3, syn0))
# test3s2 = nonlin(np.dot(test3s1, syn1))
# print("Output of JOY with ANGER data: ")
# print(test3s2)
#
# nontrainingdata4 = np.array([ [0.78], [0.8], [0.92], [0.98] ])
# test4s1 = nonlin(np.dot(nontrainingdata4, syn0))
# test4s2 = nonlin(np.dot(test4s1, syn1))
# print("Output of JOY with with FEAR data: ")
# print(test4s2)

def joy(annoutput):
    if annoutput == 1:
        print("TEST: It's JOY!")

    else:
        print("TEST: It's not JOY")
