"""
@author:Wang Xinsheng
@File:main.py
@description:...
@time:2020-12-09 21:12
"""

import pandas as pd
import tensorflow as tf
from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 读取数据
df = pd.read_excel('boston_housing_data.xlsx',header=0)
print(df)
# 查看数据的描述 平均值 最大值 最小值 数量
# print(df.describe())
# dataframe中获取np.array值
df = df.values

print('dfshape',df.shape)
# 获取0-12列数据 共 13列
x_data = df[:,:13]
for i in range(13):
    x_data[:,i] = (x_data[:,i] - x_data[:,i].min()) / (x_data[:,i].max() - x_data[:,i].min())
print(x_data)
y_data = df[:,13]
train_num = 300
vaild_num = 100 # 验证集
test_num = len(x_data) - train_num - vaild_num



x_train = x_data[:train_num]
y_train = y_data[:train_num]
x_vaild = x_data[train_num:train_num+vaild_num]
y_vaild = y_data[train_num:train_num+vaild_num]
x_test = x_data[train_num+vaild_num:train_num+vaild_num+test_num]
y_test = y_data[train_num+vaild_num:train_num+vaild_num+test_num]

# 类型转换
x_train = tf.cast(x_train,tf.float32)
x_vaild = tf.cast(x_vaild,tf.float32)
x_test = tf.cast(x_test,tf.float32)

def model(x,w,b):
    '''
    定义模型
    :param x:
    :param w:
    :param b:
    :return:
    '''
    # print("x: {} w:{} b:{} ".format(x.dtpye,w.dtype,b.dtype))
    return tf.matmul(x,w)+b

# 均值为0，标准差为1
W = tf.Variable(tf.random.normal([13,1],mean=0.0,stddev=1.0,dtype=tf.float32))
B = tf.Variable(tf.zeros(1),dtype=tf.float32)
print(W)
print(B)
print(x_data.shape)

train_epochs = 200
learning_rate = 0.03
batch_size = 5

def loss(x,y,w,b):
    '''
    定义损失函数
    :param x:
    :param y:
    :param w:
    :param b:
    :return:
    '''
    err = model(x,w,b) - y
    squared_err = tf.square(err)
    return tf.reduce_mean(squared_err)
def loss2(x,y,w,b):
    '''
    定义损失函数
    :param x:
    :param y:
    :param w:
    :param b:
    :return:
    '''
    err = model(x,w,b) - y
    squared_err = tf.square(err)
    return squared_err

def grad(x,y,w,b):
    '''

    :param x:
    :param y:
    :param w:
    :param b:
    :return: 返回梯度向量
    '''
    with tf.GradientTape() as tape:
        loss_ = loss(x,y,w,b)
        return tape.gradient(loss_,[w,b])

optimizer = tf.keras.optimizers.SGD(learning_rate)

loss_list_train = []
loss_list_vaild = []

total_step = train_num // batch_size


for epoch in range(train_epochs):
    for step in range(total_step):
        xs = x_train[step*batch_size:(step+1)*batch_size,:]
        ys = y_train[step*batch_size:(step+1)*batch_size]
        # 计算梯度
        grads = grad(xs,ys,W,B)

        # 优化器根据梯度自动调整w和b
        optimizer.apply_gradients(zip(grads,[W,B]))
    loss_train = loss(x_train,y_train,W,B).numpy()

    loss_vaild = loss(x_vaild,y_vaild,W,B).numpy()
    loss_list_train.append(loss_train)
    loss_list_vaild.append(loss_vaild)
    print("epoch={},train_loss:{},vaild_loss:{}".format(epoch+1,loss_train,loss_vaild))
plt.plot(loss_list_train,'red',label='train_loss')
plt.plot(loss_list_vaild,'blue',label='vaild_loss')
# 设置图例位置
plt.legend(loc=1)
plt.show()

loss = loss(x_test,y_test,W,B).numpy()
print(loss)


