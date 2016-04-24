
#  Pattern is recognized when it begins with 1 0 1

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [1, 0, 1, 0, 0, 0],
               [1, 0, 1, 1, 1, 1],
               [1, 1, 1, 0, 0, 1],
               [0, 0, 0, 1, 1, 1]])

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
syn2 = 2*np.random.random((4, 4)) - 1
syn3 = 2*np.random.random((4, 1)) - 1


# ################################ layer1
for j in range(60000):

    # forward propagation 1 (ANN guesses the answers)
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))
    l4 = nonlin(np.dot(l3, syn3))

    # error calculation
    l4_error = y - l4

    # if (j% 10000) == 0:
    #     print("Error:" + str(np.mean(np.abs(l2_error))))



# in what direction is the target value?
# were we really sure? if so, don't change too much.
    l4_delta = l4_error * nonlin(l4, deriv=True)

# how much did each l3 value contribute to l3 error (according to the weights)?
    l3_error = l4_delta.dot(syn3.T)




# in what direction is the target value?
# were we really sure? if so, don't change too much.
    l3_delta = l3_error*nonlin(l3, deriv=True)

# how much did each l2 value contribute to l3 error (according to the weights)?
    l2_error = l3_delta.dot(syn2.T)

# in what direction is the target value?
# were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2, deriv=True)

# how much did each l1 value contribute to l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

# in what direction is the target l1?
# were we really sure? if so don't change much
    l3_delta = l3_error * nonlin(l3, deriv=True)

# in what direction is the target l1?
# were we really sure? if so don't change much
    l2_delta = l2_error * nonlin(l2, deriv=True)

# in what direction is the target l1?
# were we really sure? if so don't change much
    l1_delta = l1_error*nonlin(l1, deriv=True)


# update weights
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn2 += l2.T.dot(l3_delta)
    syn3 += l3.T.dot(l4_delta)

print("Output of l4 in SADNESS after training: ")
print(l4)


# test the ANN with nontraining data
nontrainingdata1 = np.array([ [1, 0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 0, 1],
                              [0, 1, 0, 1, 1, 1],
                              [0, 0, 1, 0, 1, 0]])
test1s1 = nonlin(np.dot(nontrainingdata1, syn0))
test1s2 = nonlin(np.dot(test1s1, syn1))
test1s3 = nonlin(np.dot(test1s2, syn2))
test1s4 = nonlin(np.dot(test1s3, syn3))
print("Output of SADNESS with SADNESS data: ")
print(test1s4)

# nontrainingdata2 = np.array([ [0.26], [0.30], [0.41], [0.44] ])
# test2s1 = nonlin(np.dot(nontrainingdata2, syn0))
# test2s2 = nonlin(np.dot(test2s1, syn1))
# print("Output of SADNESS with SADNESS data: ")
# print(test2s2)
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