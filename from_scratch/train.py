# 신경망 학습

## 학습: 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것
## 손실함수(loss function): 성능의 나쁨을 나타내는 지표, 손실함수의 결과값을 가장 작게 만드는 가중치 매개변수를 찾는 것이 학습의 목표

""""""
import numpy as np

# 오차제곱합(SSE, Sum of Squares for Error)
# y: 추정값, t: 정답값
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t)**2)

""""""

# example)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # one-hot encoding | 정답값

# inference
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
correct = sum_squares_error(np.array(y1), np.array(t))
print(correct)   # >>> 0.09750000000000003

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] 
wrong = sum_squares_error(np.array(y2), np.array(t))
print(wrong)   # >>> 0.5975




""""""

# 교차 엔트로피 오차(CEE, Cross Entropy Error)
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

""""""

# example)
cross_correct = cross_entropy_error(np.array(y1), np.array(t))
print(cross_correct)   # >>> 0.510825457099338

cross_wrong = cross_entropy_error(np.array(y2), np.array(t))
print(cross_wrong)   # >>> 2.302584092994546


