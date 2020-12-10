"""
@author:Wang Xinsheng
@File:main.py
@description:...
@time:2020-11-23 22:06
"""
import pandas as pd
import matplotlib.pyplot as plt

seasons = ['SP','SU','FA','WI',]
start = 2000
end = 2020


file_content = pd.read_csv("Concentration - Seasonal.csv")
# print(file_content)
# file_content['']
SO2_CONC = []
SO4_CONC = []
TNO3_CONC  = []
NH4_CONC = []
years = []
index = []
count = 0
for y in range(2000,2021):
    years.append(y)
    for s in seasons:
        temp = file_content[(file_content.YEAR == y)  & (file_content.SEASON == s)]
        # 保留三位小数
        SO2_CONC.append(round(temp['SO2_CONC'].mean(),3))
        SO4_CONC.append(round(temp['SO4_CONC'].mean(),3))
        TNO3_CONC.append(round(temp['TNO3_CONC'].mean(),3))
        NH4_CONC.append(round(temp['NH4_CONC'].mean(),3))
        index.append(count)
        count += 1


fig = plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title('requests')
# alpha 透明度
plt.scatter(index,NH4_CONC,c='blue',alpha=0.2,marker='.',label='NH4_CONC')
plt.grid(True)
plt.legend(loc='best')
plt.show()
# print(temp["SO2_CONC"].mean())
# print(temp.mean().values)
# print(temp.mean().index)

print(SO2_CONC)
# 打印csv信息
# print(file_content.info())
# print(file_content.head(6))