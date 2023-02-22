# Numerical Differentiation

## Differentiation
def numerical_diff(f, x):
    h = 1e-50   # 아주 작은 값
    return (f(x + h) - f(x)) / h   # [problem] rounding error

def numerical_diff(f, x):
    h = 1e-4   # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)   # 중심차분 or 중앙차분