#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu
 
#葡萄酒质量预测模型
import sys
sys.path.append('./')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn import metrics
 
#解决中文显示问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
 
#拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)
 
#导入数据
path1 = "D:/Alpha/AI Blueprint plan/datas/winequality-red.csv"
df1 = pd.read_csv(path1, sep=";")
df1['type'] = 1
 
path2 = "datas/winequality-white.csv"
df2 = pd.read_csv(path2, sep=";")
df2['type'] = 2
 
df = pd.concat([df1,df2], axis=0)
 
names = ["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]
quality = "quality"
#print(df.head(5))
 
#对异常数据进行清除
new_df = df.replace('?', np.nan)
datas = new_df.dropna(how = 'any')
print ("原始数据条数:%d；异常数据处理后数据条数:%d；异常数据条数:%d" % (len(df), len(datas), len(df) - len(datas)))
 
#数据提取与数据分割
X = datas[names]
Y = datas[quality]
 
#划分训练集与测试集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
print ("训练数据条数:%d；数据特征个数:%d；测试数据条数:%d" % (X_train.shape[0], X_train.shape[1], X_test.shape[0]))
 
#对数据的训练集进行标准化
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
 
#构建并训练模型
lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                          multi_class='multinomial', penalty='l2', solver='lbfgs')
lr.fit(X_train, Y_train)
 
##模型效果获取
r = lr.score(X_train, Y_train)
print ("R值：", r)
print ("特征稀疏化比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print ("参数：",lr.coef_)
print ("截距：",lr.intercept_)
 
#预测
X_test = mms.transform(X_test)
Y_predict = lr.predict(X_test)
 
#画图对预测值和实际值进行比较
x_len = range(len(X_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(-1,11)
plt.plot(x_len, Y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize = 12, zorder=2, label=u'预测值,$R^2$=%.3f' % lr.score(X_train, Y_train))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'葡萄酒质量', fontsize=18)
plt.title(u'葡萄酒质量预测统计', fontsize=20)
plt.savefig("葡萄酒质量预测统计.png")
plt.show()
 
# #运行结果：
# 原始数据条数:6497；异常数据处理后数据条数:6497；异常数据条数:0
# 训练数据条数:4872；数据特征个数:12；测试数据条数:1625
# R值： 0.549466338259
# 特征稀疏化比率：0.00%
# 参数： [[ 0.97934119  2.16608604 -0.41710039 -0.49330657  0.90621136  1.44813439
#    0.75463562  0.2311527   0.01015772 -0.69598672 -0.71473577 -0.2907567 ]
#  [ 0.62487587  5.11612885 -0.38168837 -2.16145905  1.21149753 -3.71928146
#   -1.45623362  1.34125165  0.33725355 -0.86655787 -2.7469681   2.02850838]
#  [-1.73828753  1.96024965  0.48775556 -1.91223567  0.64365084 -1.67821019
#    2.20322661  1.49086179 -1.36192671 -2.2337436  -5.01452059 -0.75501299]
#  [-1.19975858 -2.60860814 -0.34557812  0.17579494 -0.04388969  0.81453743
#   -0.28250319  0.51716692 -0.67756552  0.18480087  0.01838834 -0.71392084]
#  [ 1.15641271 -4.6636028  -0.30902483  2.21225522 -2.00298042  1.66691445
#   -1.02831849 -2.15017982  0.80529532  2.68270545  3.36326129 -0.73635195]
#  [-0.07892353 -1.82724304  0.69405191  2.07681409 -0.6247279   1.49244742
#   -0.16115782 -1.3671237   0.72694885  1.06878382  4.68718155  0.04669067]
#  [ 0.25633987 -0.14301056  0.27158425  0.10213705 -0.08976172 -0.02454203
#   -0.02964911 -0.06312954  0.15983679 -0.14000195  0.40739327  0.42084343]]
# 截距： [-2.34176729 -1.1649153   4.91027564  4.3206539   1.30164164 -2.25841567
#  -4.76747291]
