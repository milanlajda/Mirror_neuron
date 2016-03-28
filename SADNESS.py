import numpy as np


# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [1.5], [1.8], [2.1], [2.4] ])

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

print("Output from SADNESS after training: ")
print (l1)







def sadness(annoutput):
    if annoutput == 2:
        print("TEST: It's SADNESS!")

    else:
        print("TEST: It's not SADNESS")