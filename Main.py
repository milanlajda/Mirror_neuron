from ENCAPSULATION import feedthedata

import numpy as np

def main():

# test data that ANN should recognize as JOY
    testj1 = np.array([0, 1, 0, 0, 0, 0])
    testj2 = np.array([0, 1, 0, 0, 0, 1])
    testj3 = np.array([0, 1, 0, 1, 1, 1])

# test data that ANN should recognize as ANGER
    testa1 = np.array([1, 1, 1, 0, 0, 0])
    testa2 = np.array([1, 1, 1, 0, 1, 0])
    testa3 = np.array([1, 1, 1, 0, 0, 1])

# test data that ANN should recognize as FEAR
    testf1 = np.array([0, 0, 0, 0, 0, 0])
    testf2 = np.array([0, 0, 0, 1, 1, 1])
    testf3 = np.array([0, 0, 0, 0, 1, 0])

# run the test for JOY
    feedthedata(testj1)
    feedthedata(testj2)
    feedthedata(testj3)

# run the test for ANGER
    feedthedata(testa1)
    feedthedata(testa2)
    feedthedata(testa3)

# run the test for FEAR
    feedthedata(testf1)
    feedthedata(testf2)
    feedthedata(testf3)





if __name__ == "__main__":
    main()
