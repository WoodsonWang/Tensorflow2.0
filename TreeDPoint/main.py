"""
@author:Wang Xinsheng
@File:main.py
@description:...
@time:2020-11-16 13:01
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 我们先随机生成一些数字,作为点云输入,为了减少物体尺度的问题,
#通常会将点云缩到半径为1的球体中
#为了方便起见,LZ把batch_size改成1
point_cloud = np.random.rand(1, 1024, 100000) - np.random.rand(1, 1024, 100000)

#画出3d点云
def pyplot_draw_point_cloud(points, output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # savefig(output_filename)
# pyplot_draw_point_cloud(point_cloud)
class point:
    def __init__(self,z,x,y):
        self.z = z
        self.x = x
        self.y = y


with open("data.txt") as f:
    for l in f.readlines():
        points = l.strip(',').split(',')

fig = plt.figure()
ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')

x = []
y = []
z = []
for p in points:
    p = p.split(' ')
    print(p)
    x.append(float(p[1]))
    y.append(float(p[2]))
    z.append(float(p[0]))


ax.scatter(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
print(len(points))
