"""
@author:Wang Xinsheng
@File:one.py
@description:...
@time:2020-10-12 11:13
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 设置随机数种子
np.random.seed(5)

# 生成等差数列，一共100个点,取值在-1到1之间
x_data = np.linspace(-1,1,100)
# randn 是标准正态分布，以0为均值，以1为标准差的正态分布
# randn(D1,D2....)   D1 表示第一维度的大小
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4

print(y_data)
plt.scatter(x_data,y_data,)
plt.plot(x_data,2.0*x_data+1.0, color='red', linewidth=3)
# 显示画的图
plt.show()

# 定义模型
def model(x,w,b):
    return tf.multiply(x,w)+b

w = tf.Variable(np.random.randn(),tf.float32)
b = tf.Variable(0.0,tf.float32)

# 定义损失函数
def loss(x,y,w,b):
    err = model(x,w,b) - y
    squared_err = tf.square(err)
    return tf.reduce_mean(squared_err)

training_epochs = 10
learning_rate = 0.01

# 计算样本数据在[w,b] 点上的梯度 2.0 才有
def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_ = loss(x,y,w,b)
    return tape.gradient(loss_,[w,b])

loss_list = []
display_step = 10
step =0
for epoch in range(training_epochs):
    for xs,ys in zip(x_data,y_data):
        loss_ = loss(xs,ys,w,b)
        loss_list.append(loss_)

        delta_w,delta_b = grad(xs,ys,w,b)
        change_w = learning_rate * delta_w
        change_b = learning_rate * delta_b
        w.assign_sub(change_w)
        b.assign_sub(change_b)

        step = step + 1
        if step % display_step == 0:
            print("{}epoch,{}step,{}".format(epoch,step,loss_))

    plt.plot(x_data,w.numpy() * x_data + b.numpy())

plt.show()