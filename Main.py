from ENCAPSULATION import feedthedata

import numpy as np

def main():

# test data that ANN should recognize as JOY
    test1 = np.array([0, 1, 0, 0, 0, 0])

# test data that ANN should recognize as ANGER
    test2 = np.array([1, 1, 1, 0, 0, 0])

# test data that ANN should recognize as FEAR
    test3 = np.array([0, 0, 0, 0, 0, 0])

    feedthedata(test1)
    feedthedata(test2)
    feedthedata(test3)





if __name__ == "__main__":
    main()
