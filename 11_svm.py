#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu
 
#鸢尾花数据SVM分类
import sys
sys.path.append('./')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm     #svm导入
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
#解决中文显示问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
 
#数据读取
path = "ML\case\datas\iris.data"
data = pd.read_csv(path,header=None)
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
x,y = data[list(range(4))],data[4]
y = pd.Categorical(y).codes     #将文本数据进行编码，如：a b c编码为 0 1 2
x = x[[0,1]]
 
#数据分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.6,random_state=28)
 
# svm.SVC API说明：
# 功能：使用SVM分类器进行模型构建
# 参数说明：
# C: 误差项的惩罚系数，默认为1.0；一般为大于0的一个数字，C越大表示在训练过程中对于总误差的关注度越高，也就是说当C越大的时候，对于训练集的表现会越好，
# 但是有可能引发过度拟合的问题(overfiting)
# kernel：指定SVM内部函数的类型，可选值：linear、poly、rbf、sigmoid、precomputed(基本不用，有前提要求，要求特征属性数目和样本数目一样)；默认是rbf；
# degree：当使用多项式函数作为svm内部的函数的时候，给定多项式的项数，默认为3
# gamma：当SVM内部使用poly、rbf、sigmoid的时候，核函数的系数值，当默认值为auto的时候，实际系数为1/n_features
# coef0: 当核函数为poly或者sigmoid的时候，给定的独立系数，默认为0
# probability：是否启用概率估计，默认不启动，不太建议启动
# shrinking：是否开启收缩启发式计算，默认为True
# tol: 模型构建收敛参数，当模型的的误差变化率小于该值的时候，结束模型构建过程，默认值:1e-3
# cache_size：在模型构建过程中，缓存数据的最大内存大小，默认为空，单位MB
# class_weight：给定各个类别的权重，默认为空
# max_iter：最大迭代次数，默认-1表示不限制
# decision_function_shape: 决策函数，可选值：ovo和ovr，默认为None；推荐使用ovr；（1.7以上版本才有）
 
#数据svm分类器构建
clf = svm.SVC(C=1,kernel="linear")
#模型训练
clf.fit(x_train,y_train)
 
#计算模型的准确率
print("score:",clf.score(x_train,y_train))
print("训练集准确率:",accuracy_score(y_train,clf.predict(x_train)))
print("score:",clf.score(x_test,y_test))
print("测试集准确率:",accuracy_score(y_test,clf.predict(x_test)))
 
#计算决策函数的结构值与预测值
#decision_function 计算的是样本x到各个分割平面的距离<也就是决策函数的值>
print("decision_function:\n",clf.decision_function(x_train))
print("npredict\n",clf.predict(x_train))
 
#-画图
N = 500
x1_min,x2_min = x.min()
x1_max,x2_max = x.max()
t1 = np.linspace(x1_min,x1_max,N)
t2 = np.linspace(x2_min,x2_max,N)
x1,x2 = np.meshgrid(t1,t2)      #生成网格采样点
 
grid_show = np.dstack((x1.flat,x2.flat))[0]     #测试点
grid_hat = clf.predict(grid_show)       #预测分类值
grid_hat = grid_hat.reshape(x1.shape)   #使之与输入的形状相同
 
cm_light = mpl.colors.ListedColormap(["#A0FFA0","#FFA0A0","#A0A0FF"])
cm_dark = mpl.colors.ListedColormap(["g","r","b"])
 
plt.figure(facecolor="w")
#区域图
plt.pcolormesh(x1,x2,grid_hat,cmap=cm_light)
#所有样本点
plt.scatter(x[0],x[1],c=y,edgecolors="k",s=50,cmap=cm_dark)
#测试数据集
plt.scatter(x_test[0],x_test[1],s=120,facecolor="none",zorder=10)   #圈中测试集样本
plt.xlabel(iris_feature[0],fontsize=13)
plt.ylabel(iris_feature[1],fontsize=13)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title("鸢尾花SVM特征分类",fontsize=16)
plt.grid(b=True,ls=":")
plt.tight_layout(pad=1.5)
plt.savefig("ML\case\鸢尾花SVM特征分类.png")
plt.show()
 
# #运行结果：
# score: 0.85
# 训练集准确率: 0.85
# score: 0.711111111111
# 测试集准确率: 0.711111111111
# decision_function:
#  [[-0.1735954   1.07024224  2.10335316]
#  [-0.24032342  2.13157018  1.10875323]
#  [ 0.99570201  2.0769661  -0.0726681 ]
#  [-0.16985611  1.00818979  2.16166632]
#  [-0.19326418  1.07479258  2.1184716 ]
#  [ 2.1722349  -0.11684293  0.94460803]
#  [-0.07947693  0.95668703  2.1227899 ]
#  [ 2.07787335 -0.0797157   1.00184235]
#  [-0.31924164  1.07334357  2.24589807]
#  [-0.12255379  1.02784016  2.09471363]
#  [-0.43725436  1.10064561  2.33660874]
#  [-0.20895059  1.09371845  2.11523215]
#  [-0.37426563  1.10137012  2.27289551]
#  [-0.22488009  1.03621634  2.18866375]
#  [-0.10288501  1.02328982  2.07959518]
#  [ 2.20385081  0.92173331 -0.12558412]
#  [ 2.09380285  0.97778641 -0.07158926]
#  [ 2.18418203  0.92628365 -0.11046568]
#  [ 2.12493258 -0.1364933   1.01156072]
#  [-0.22861938  1.09826879  2.13035059]
#  [-0.26023529  1.05969255  2.20054274]
#  [ 2.15256612 -0.11229259  0.95972647]
#  [-0.19724656  1.06041705  2.13682951]
#  [-0.08321622  1.01873948  2.06447674]
#  [-0.22488009  1.03621634  2.18866375]
#  [-0.23658413  1.06951773  2.16706639]
#  [-0.16587374  1.02256532  2.14330842]
#  [-0.09865955  2.11409333  0.98456622]
#  [-0.20122893  1.04604152  2.15518741]
#  [ 1.01935317  2.08679128 -0.10614445]
#  [-0.12255379  1.02784016  2.09471363]
#  [-0.11060667  2.07096674  1.03963992]
#  [-0.24032342  2.13157018  1.10875323]
#  [-0.31176307  0.94923867  2.3625244 ]
#  [-0.15790899  1.05131637  2.10659262]
#  [-0.29160811  1.09754428  2.19406383]
#  [ 2.19962535 -0.16907019  0.96944484]
#  [-0.11060667  2.07096674  1.03963992]
#  [-0.26421766  1.04531702  2.21890064]
#  [ 2.18792131 -0.1357688   0.94784748]
#  [ 2.15654849 -0.09791706  0.94136857]
#  [-0.24479196  0.9643387   2.28045326]
#  [ 2.11721092 -0.08881638  0.97160546]
#  [-0.20122893  1.04604152  2.15518741]
#  [-0.26771387  2.18379745  1.08391642]
#  [-0.06306126  2.1670451   0.89601617]
#  [ 2.13314042  0.96868573 -0.10182615]
#  [-0.5         1.17634909  2.32365091]
#  [-0.33891043  1.07789391  2.26101652]
#  [ 2.15654849 -0.09791706  0.94136857]
#  [-0.0712691   2.06186606  1.00940303]
#  [ 2.19612915  0.96941024 -0.16553939]
#  [ 2.13314042  0.96868573 -0.10182615]
#  [-0.22488009  1.03621634  2.18866375]
#  [-0.27169624  2.16942192  1.10227432]
#  [ 2.1881644   0.94065918 -0.12882358]
#  [ 2.2270158  -0.22129746  0.99428166]
#  [ 2.26659645 -0.15397016  0.8873737 ]
#  [ 2.15630541 -0.17434504  1.01803963]
#  [ 2.11347163  0.97323607 -0.08670771]]
# npredict
#  [2 1 1 2 2 0 2 0 2 2 2 2 2 2 2 0 0 0 0 2 2 0 2 2 2 2 2 1 2 1 2 1 1 2 2 2 0
#  1 2 0 0 2 0 2 1 1 0 2 2 0 1 0 0 2 1 0 0 0 0 0]
