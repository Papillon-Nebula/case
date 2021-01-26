#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu
 
#线性回归——家庭用电预测（功率与电压之间的关系）
 
#导入模块
import sys
sys.path.append('./')
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
 
#导入数据
path = "../../datas/household_power_consumption_1000.txt"
data = pd.read_csv(path,sep=";")
 
#iloc进行行列切片只能用数字下标，取出X的原始值（所有行与二、三列的表示功率的数据）
x = data.iloc[:,2:4]
y = data.iloc[:,5]      #取出Y的数据（电流）
 
#划分训练集与测试集，random_state是随机数发生器使用的种子
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
 
#对训练集和测试集进行标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
 
#建立线性模型
lr = LinearRegression()
lr.fit(x_train,y_train)     #训练
print("预测的决定系数R平方:",lr.score(x_train,y_train))
print("线性回归的估计系数:",lr.coef_)     #打印线性回归的估计系数
print("线性模型的独立项:",lr.intercept_)    #打印线性模型的独立项
 
y_predict = lr.predict(x_test)  #预测
# print(y_predict)
 
#模型效果判断
mse = np.average((y_predict-np.array(y_test))**2)
rmse = np.sqrt(mse)
print("均方误差平方和：",mse)
print("均方误差平方和的平方根：",rmse)
 
#模型的保存与持久化
from sklearn.externals import joblib
 
joblib.dump(ss,"PI_data_ss.model")      #将标准化模型保存
joblib.dump(lr,"PI_data_lr.model")      #将训练后的线性模型保存
 
joblib.load("PI_data_ss.model")         #加载模型,会保存该model文件
joblib.load("PI_data_lr.model")         #加载模型
 
#预测值和实际值画图比较
 
#解决中文问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
 
p = np.arange(len(x_test))
plt.figure(facecolor="w")   #创建画布，facecolor为背景色，w是白色
plt.plot(p,y_test,"r-",linewidth = 2,label = "真实值")
plt.plot(p,y_predict,"g-",linewidth = 2,label = "预测值")
plt.legend(loc = "upper right")     #显示图例，设置图例的位置
plt.title("线性回归预测功率和电流之间的关系",fontsize = 20)
plt.grid(b=True)
plt.savefig("线性回归预测功率和电流之间的关系.png")
plt.show()
 
# #运行结果：
# 预测的决定系数R平方: 0.990719383392
# 线性回归的估计系数: [ 5.12959849  0.0589354 ]
# 线性模型的独立项: 10.3485714286
# 均方误差平方和： 0.193026891251
# 均方误差平方和的平方根： 0.439348257366
