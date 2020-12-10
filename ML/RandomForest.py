# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:46:07 2019

@author: zhangmingzheng
"""
import time
import gdal
import os
start = time.perf_counter()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt
data1 = np.loadtxt(open("disor.csv","rb"),delimiter=",",skiprows=0)
data2 = np.loadtxt(open("healthor.csv","rb"),delimiter=",",skiprows=0)
dis1=round(data1.shape[0]*0.7)
hea1=round(data2.shape[0]*0.7)
#print(data2.shape[0]) 
#print(hea1)
arr_1 = np.random.choice(data1.shape[0],dis1,replace=False)
arr_2 = np.arange(data1.shape[0])
arr_2 = np.delete(arr_2,arr_1)
arr_3 = np.random.choice(data2.shape[0],hea1,replace=False)
arr_4 = np.arange(data2.shape[0])
arr_4 = np.delete(arr_4,arr_3,axis=0)

#print(arr_4.shape[0])

#print(data1[rand_arr,:])
a=data1[arr_1,:]
a1=a[:,[0,1,2,3]]
a2=a[:,[5]]
b=data2[arr_3,:]
b1=b[:,[0,1,2,3]]
b2=b[:,[5]]
x_train=np.concatenate((a1,b1),axis=0)
y_train=np.concatenate((a2,b2),axis=0)
c=data1[arr_2,:]
c1=c[:,[0,1,2,3]]
c2=c[:,[5]]
d=data2[arr_4,:]
d1=d[:,[0,1,2,3]]
d2=d[:,[5]]
x_test=np.concatenate((c1,d1),axis=0)
y_test=(np.concatenate((c2,d2),axis=0)).ravel()

#X=data[:,[0,1,2,3,4]]
#Y=data[:,[5]]
#建立模型，并进行模型训练。
clf=RandomForestClassifier(random_state=15)
clf.fit(x_train,y_train.ravel())
print(x_train[1,:])
#获得变量权重。
X_importance=clf.feature_importances_

#模型预测。
y_pred=clf.predict(x_test)
#y_test=np.array(y_test)
#print(y_test)
#print(y_pred)
reslut=np.abs(y_test-y_pred)
#print(sum(reslut))
#print(reslut.shape[0])
pred=100-sum(reslut)*100.0/(reslut.shape[0])
print(pred)
end = time.perf_counter()
t=end-start
print("Runtime is ：",t)
class GRID:
  #读图像文件
  def read_img(self,filename):
    dataset=gdal.Open(filename)    #打开文件

    im_width = dataset.RasterXSize  #栅格矩阵的列数
    im_height = dataset.RasterYSize  #栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵

    del dataset 
    return im_proj,im_geotrans,im_data 

  #写文件，以写成tif为例
  def write_img(self,filename,im_proj,im_geotrans,im_data):
    #gdal数据类型包括
    #gdal.GDT_Byte, 
    #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    #gdal.GDT_Float32, gdal.GDT_Float64

    #判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
      datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
      datatype = gdal.GDT_UInt16
    else:
      datatype = gdal.GDT_Float32

    #判读数组维数
    if len(im_data.shape) == 3:
      im_bands, im_height, im_width = im_data.shape
    else:
      im_bands, (im_height, im_width) = 1,im_data.shape 

    #创建文件
    driver = gdal.GetDriverByName("GTiff")      #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)       #写入仿射变换参数
    dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data) #写入数组数据
    else:
      for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

if __name__ == "__main__":
  os.chdir(r'J:\2019paper\data')            #切换路径到待处理图像所在文件夹
  run = GRID()
  proj,geotrans,data = run.read_img('multy_m.tif')
  m=data.shape[1]
  n=data.shape[2]
  print(m)
  print(n)
  in_data=np.zeros((m,n))
  #data = data.astype(np.float)
 # aa=np.array(data[:,555,:]).T
  for i in range(m):
      print(i)
      y_pred=clf.predict(np.array(data[:,i,:]).T)
 #     for j in range(n):
 #         if data[0,i,j]>0:
 #               y_pred=clf.predict(np.array([data[:,i,j]]).reshape(1, -1))
      in_data[i,:]=y_pred.T
 #         else:
 #               in_data[i,j]=0
 #ndvi = (data[7]-data[3])/(data[7]+data[3])             #3为近红外波段；2为红波段
  run.write_img('class_re12.tif',proj,geotrans,in_data) #写为ndvi图像
