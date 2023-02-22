# Numerical Differentiation

import numpy as np
#import matplotlib as plt
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt

## Differentiation
def numerical_diff(f, x):
    h = 1e-50   # 아주 작은 값
    return (f(x + h) - f(x)) / h   # [problem] rounding error

def numerical_diff(f, x):
    h = 1e-4   # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)   # 중심차분 or 중앙차분

""""""

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)   # 0 ~ 20미만까지 0.1 간격
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

""""""

numerical_diff(function_1, 5)   # 0.1999999999990898
numerical_diff(function_1, 10)  # 0.2999999999986347

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

tf = tangent_line(function_1, 10)
y3 = tf(x)

plt.plot(x, y)
plt.plot(x, y3)
plt.show()