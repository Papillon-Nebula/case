#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu
 
#Adaboost算法
 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
 
#解决中文显示问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
 
#创建数据
#生成2维正态分布，生成的数据按分位数分为两类，200个样本,2个样本特征，协方差系数为2
X1,y1 = make_gaussian_quantiles(cov=2,n_samples=200,n_features=2,
                                n_classes=2,random_state=1) #创建符合高斯分布的数据集
X2,y2 = make_gaussian_quantiles(mean=(3,3),cov=1.5,n_samples=300,n_features=2,
                                n_classes=2,random_state=1)
#将两组数据合成一组数据
X = np.concatenate((X1,X2))
y = np.concatenate((y1,-y2+1))
 
#构建adaboost模型
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME.R",n_estimators=200)
 
#数据量大时，可以增加内部分类器的max_depth(树深)，也可不限制树深，树深的范围为：10-100
#数据量小时，一般可以设置树深较小或者n_estimators较小
#n_estimators:迭代次数或最大弱分类器数
#base_estimator:DecisionTreeClassifier，选择弱分类器，默认为CART树
#algorithm：SAMME和SAMME.R，运算规则，后者是优化算法，以概率调整权重，迭代，需要有能计算概率的分类器支持
#learning_rate：0<v<=1,默认为1,正则项 衰减指数
#loss：误差计算公式，有线性‘linear’,平方‘square’和指数'exponential’三种选择,一般用linear足够
 
#训练
bdt.fit(X,y)
 
plot_step = 0.02
x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
#meshgrid的作用：生成网格型数据
xx,yy = np.meshgrid(np.arange(x_min,x_max,plot_step),
                    np.arange(y_min,y_max,plot_step))
 
#预测
# np.c_  按照列来组合数组
Z = bdt.predict(np.c_[xx.ravel(),yy.ravel()])
#设置维度
Z = Z.reshape(xx.shape)
 
#画图
plot_coloes = "br"
class_names = "AB"
 
plt.figure(figsize=(10,5),facecolor="w")
#局部子图
plt.subplot(1,2,1)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)
for i,n,c in zip(range(2),class_names,plot_coloes):
    idx = np.where(y == i)
    plt.scatter(X[idx,0],X[idx,1],c=c,cmap=plt.cm.Paired,label=u"类别%s"%n)
 
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.legend(loc="upper right")
plt.xlabel("x")
plt.ylabel("y")
plt.title(u"Adaboost分类结果,正确率为:%.2f%%"%(bdt.score(X,y)*100))
plt.savefig("ML\case\Adaboost分类结果.png")
 
#获取决策函数的数值
twoclass_out = bdt.decision_function(X)
#获取范围
plot_range = (twoclass_out.min(),twoclass_out.max())
plt.subplot(1,2,2)
for i,n,c in zip(range(2),class_names,plot_coloes):
#直方图
    plt.hist(twoclass_out[y==i],bins=20,range=plot_range,
             facecolor=c,label=u"类别%s"%n,alpha=.5)
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,y2*1.2))
plt.legend(loc="upper right")
plt.xlabel(u"决策函数值")
plt.ylabel(u"样本数")
plt.title(u"Adaboost的决策值")
plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.savefig("ML\case\Adaboost的决策值.png")
plt.show()
