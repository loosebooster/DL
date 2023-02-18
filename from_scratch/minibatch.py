# MINI-BATCH

## 미니배치: 데이터 일부를 추려 전체의 근사치로 이용, 손실함수를 계산하기 위해 훈련 데이터로부터 골라 학습하는 일부분

""""""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)   # (60000, 784)
print(t_train.shape)   # (60000m 10)

""""""

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

""""""

# sample)
np.random.choice(60000, 10)   # >>> array([12576,  6644, 53707, 10407, 56193, 57714, 19616, 12960,  1129, 13996])