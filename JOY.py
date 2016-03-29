import numpy as np


# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [0], [0.2], [0.8], [1.4] ])

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

print("Output of JOY after training: ")
print (l1)


# test the ANN with nontraining data
nontrainingdata1 = np.array([ [0.25], [0.4], [1.15], [1.3] ])
test = nonlin(np.dot(nontrainingdata1, syn0))
print("Output of JOY with new data between 0 and 1.4: ")
print(test)

nontrainingdata2 = np.array([ [2], [3], [4], [5] ])
test2 = nonlin(np.dot(nontrainingdata2, syn0))
print("Output of JOY with new data > 1.4: ")
print(test2)

def joy(annoutput):
    if annoutput == 1:
        print("TEST: It's JOY!")

    else:
        print("TEST: It's not JOY")
