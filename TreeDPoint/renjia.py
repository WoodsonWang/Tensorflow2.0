import django
import requests
from ftplib import FTP
import ftplib
import pymysql
import requests_ftp
import re
import os
import chardet

# 对于每一列 第9个元素为访问的下一个文件夹名
List_size_of_file = 8


# requests_ftp.monkeypatch_session()
# s = requests.Session()
# res = s.list(url)
# res.encoding = 'GB2312'
# print(res.text)

# temp = "" + str(text)
# temp_list = temp.split('\r\n')
# print(temp_list[2])
# t = re.split(r'\s+', temp_list[2])
# print(t[8])
# print(len(temp_list))
# 将爬取的数组钻华为字符串进行第一次切分 切分后字符串数组长度会多1，最后一个元素为空
def Cut_of_List(List, num):
    List_first = "" + str(List)
    List_Second = List_first.split('\r\n')
    List_Second.pop()
    List_temp = []
    List_t = []
    for words in List_Second:
        if words[0] != 'd':
            List_temp.append(words)
    for wor in List_temp:
        List_Second.remove(wor)
    for word in List_Second:
        print(word)
    if len(List_Second) != 0:
        # 对每一列进行正则表达式切分，以1个至多个空格为切分依据
        for i in range(len(List_Second)):
            temp = re.split(r'\s+', List_Second[i])
            try:
                if temp[List_size_of_file + 1]:
                    List_t.append(temp[List_size_of_file] + ' ' + temp[List_size_of_file + 1])
            except:
                List_t.append(temp[List_size_of_file])

        # print(List_t)
        for w in List_t:
            print(w)
        print(len(List_Second))
        return List_t
    else:
        return ''


# requests_ftp爬取
def Requests_of_ftp(Url):
    try:
        s = requests.Session()
        r = s.get(Url, timeout=2)
        res = s.list(Url)
        res.encoding = 'utf-8'
        # print(res.text)
        s.close()
        return res.text
    except:
        return ''


# 参数 该目录地址url
# 返回 该目录下文件夹列表以及文件夹数目
def ftp_1(url):
    requests_ftp.monkeypatch_session()
    try:
        text = Requests_of_ftp(url)
    except:
        print(url + ' ' + "error")
    else:
        print(url + ' ' + "success")
        filename1 = Cut_of_List(text, 0)
    return filename1


def ftp_all(Url, add, file):
    List = ftp_1(Url)
    file.write(add)
    file.write('\n')
    for words in List:
        if not List:
            break
        if add == '/':
            ftp_all(Url + '/' + words, add + words, file)
        else:
            ftp_all(Url + '/' + words, add + '/' + words, file)




if __name__ == '__main__':
    f = open("test.txt", 'w')
    url = "ftp://211.71.149.149"
    ftp_all(url, '/', f)
