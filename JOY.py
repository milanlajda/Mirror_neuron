import numpy as np


# sigmoid function
def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
x = np.array([0, 0.2, 0.8, 1.4])

# output dataset
y = np.array([1, 1, 1, 1, ])


def joy(annoutput):
    if annoutput == 1:
        print("TEST: It's JOY!")

    else:
        print("TEST: It's not JOY")
