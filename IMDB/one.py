"""
@author:Wang Xinsheng
@File:one.py
@description:...
@time:2020-10-10 23:12
"""

import tensorflow as tf

# 定义两个常量
node1 = tf.constant([[3,1],[2,6]],tf.int32)
# 显示类型转换
node1 = tf.cast(node1,tf.float32)
node2 = tf.constant([[4.0,1.5],[2.3,6.2]],tf.float32)
node3 = tf.add(node1,node2)

# 会自动将数据转换为指定形状的。
node4 = tf.constant([1,2,3,4,5,6],shape=(2,3))

# 打印很多东西
# tf.Tensor(
# [[ 7.   3. ]
#  [ 4.8 12.2]], shape=(2, 2), dtype=float32)
print(node3)

# 直接打印数值
print(node3.numpy())
# 指定某个元素进行打印
print(node3.numpy()[0,1])
# 输入形状
print(node3.shape)
# 输出数据类型
print(node3.dtype)

print(node4)
#