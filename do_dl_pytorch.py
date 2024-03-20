import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)


c = torch.zeros(1000)

# 标量相加
# time = 0.350......248
# for i in range(1000):
#     c[i] = a[i] + b[i]
start = time()
# 矢量相加
# time 几乎不耗时
c = b + a

"""
# 房价预测
y^(1) = x_1^(1)w_1+x_2^(1)w_2+b
...

"""

print(time() - start)
