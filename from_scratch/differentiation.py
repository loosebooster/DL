# Numerical Differentiation

## Differentiation
def numerical_diff(f, x):
    h = 1e-50   # 아주 작은 값
    return (f(x + h) - f(x)) / h   # [problem] rounding error



