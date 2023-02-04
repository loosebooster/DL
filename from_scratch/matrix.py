import numpy as np
from activation import sigmoid

# Matrix-multiplication
''' Attention to the Shape of Matrix '''
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(np.dot(A, B))

C = np.array([[1,2,3], [4,5,6]])
D = np.array([[1,2], [3,4], [5,6]])
print(np.dot(C, D))

# Matrix-multiplication in Neural Networks
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])
Y = np.dot(X, W)
print(Y)

X1 = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X1, W1) + B1
Z1 = sigmoid(A1) #activation
print(Z1)