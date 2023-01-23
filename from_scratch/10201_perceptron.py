# logic gate
# perceptron algorithm

import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.8
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.8
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

# multi-layer perceptron
def XOR(x1, x2): 
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    tmp = AND(s1, s2)
    if tmp > 0:
        return 1
    else:
        return 0
