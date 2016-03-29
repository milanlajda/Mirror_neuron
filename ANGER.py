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
               0, 0, 0, 0,
               1, 1, 1, 1,
               0, 0, 0, 0]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((1, 1)) - 1

for inter in range(10000):

    # forward propagation
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))

    # error calculation
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

# print("Output of JOY after training: ")
# print (l1)


# test the ANN with nontraining data
nontrainingdata1 = np.array([ [0.1], [0.10], [0.15], [0.22] ])
test = nonlin(np.dot(nontrainingdata1, syn0))
print("Output of ANGER with with JOY data:")
print(test)

# nontrainingdata2 = np.array([ [0.26], [0.30], [0.41], [0.44] ])
# test2 = nonlin(np.dot(nontrainingdata2, syn0))
# print("Output of ANGER with SADNESS data: ")
# print(test2)

# nontrainingdata3 = np.array([ [0.52], [0.62], [0.69], [0.72] ])
# test3 = nonlin(np.dot(nontrainingdata3, syn0))
# print("Output of ANGER with ANGER data: ")
# print(test3)
#
# nontrainingdata4 = np.array([ [0.78], [0.8], [0.92], [0.98] ])
# test4 = nonlin(np.dot(nontrainingdata4, syn0))
# print("Output of ANGER with FEAR data: ")
# print(test4)


def anger(annoutput):
    if annoutput == 3:
        print("TEST: It's ANGER!")

    else:
        print("TEST: It's not ANGER")