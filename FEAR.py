import numpy as np


# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [3.5], [3.8], [4.1], [5] ])

# output dataset
y = np.array([[1, 1, 1, 1]]).T

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

print("Output from FEAR after training: ")
print (l1)


def fear(annoutput):
    if annoutput == 4:
        print("TEST: It's FEAR!")

    else:
        print("TEST: It's not FEAR")