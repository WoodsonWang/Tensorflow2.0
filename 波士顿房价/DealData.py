"""
@author:Wang Xinsheng
@File:DealData.py
@description:...
@time:2020-12-10 20:36
"""
from openpyxl import load_workbook
import re
wb = load_workbook('boston_housing_data.xlsx')
ws = wb['Sheet1']

with open('housing.data','r') as f:
    for line in f.readlines():
        # 匹配任意非空字符
        num = re.findall(r'[\S]+',line)
        ws.append(num)
wb.save('boston_housing_data.xlsx')

