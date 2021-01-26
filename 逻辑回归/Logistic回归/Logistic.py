#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu
 
#乳腺癌分类案例
import sys
sys.path.append('./')
import sklearn
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
 
#解决中文显示问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
 
#拦截异常
warnings.filterwarnings(action='ignore',category=ConvergenceWarning)
 
#导入数据并对异常数据进行清除
path = "D:/Alpha/AI Blueprint plan/datas/breast-cancer-wisconsin.data"
names = ["id","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape"
         ,"Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin"
         ,"Normal Nucleoli","Mitoses","Class"]
 
df = pd.read_csv(path,header=None,names=names)
 
datas = df.replace("?",np.nan).dropna(how="any")    #只要列中有nan值，进行行删除操作
#print(datas.head())     #默认显示前五行
 
#数据提取与数据分割
X = datas[names[1:10]]
Y = datas[names[10]]
 
#划分训练集与测试集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=0)
 
#对数据的训练集进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)     #先拟合数据在进行标准化
 
#构建并训练模型
##  multi_class:分类方式选择参数，有"ovr(默认)"和"multinomial"两个值可选择，在二元逻辑回归中无区别
##  cv:几折交叉验证
##  solver:优化算法选择参数，当penalty为"l1"时，参数只能是"liblinear(坐标轴下降法)"
##  "lbfgs"和"cg"都是关于目标函数的二阶泰勒展开
##  当penalty为"l2"时，参数可以是"lbfgs(拟牛顿法)","newton_cg(牛顿法变种)","seg(minibactch随机平均梯度下降)"
##  维度<10000时，选择"lbfgs"法，维度>10000时，选择"cs"法比较好，显卡计算的时候，lbfgs"和"cs"都比"seg"快
##  penalty:正则化选择参数，用于解决过拟合，可选"l1","l2"
##  tol:当目标函数下降到该值是就停止，叫：容忍度，防止计算的过多
lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
re = lr.fit(X_train,Y_train)
 
#模型效果获取
r = re.score(X_train,Y_train)
print("R值(准确率):",r)
print("参数:",re.coef_)
print("截距:",re.intercept_)
print("稀疏化特征比率:%.2f%%" %(np.mean(lr.coef_.ravel()==0)*100))
print("=========sigmoid函数转化的值，即：概率p=========")
print(re.predict_proba(X_test))     #sigmoid函数转化的值，即：概率p
 
#模型的保存与持久化
from sklearn.externals import joblib
joblib.dump(ss,"logistic_ss.model")     #将标准化模型保存
joblib.dump(lr,"logistic_lr.model")     #将训练后的线性模型保存
joblib.load("logistic_ss.model")        #加载模型,会保存该model文件
joblib.load("logistic_lr.model")
 
#预测
X_test = ss.transform(X_test)       #数据标准化
Y_predict = lr.predict(X_test)      #预测
 
#画图对预测值和实际值进行比较
x = range(len(X_test))
plt.figure(figsize=(14,7),facecolor="w")
plt.ylim(0,6)
plt.plot(x,Y_test,"ro",markersize=8,zorder=3,label=u"真实值")
plt.plot(x,Y_predict,"go",markersize=14,zorder=2,label=u"预测值,$R^2$=%.3f" %lr.score(X_test,Y_test))
plt.legend(loc="upper left")
plt.xlabel(u"数据编号",fontsize=18)
plt.ylabel(u"乳癌类型",fontsize=18)
plt.title(u"Logistic算法对数据进行分类",fontsize=20)
plt.savefig("Logistic算法对数据进行分类.png")
plt.show()
 
print("=============Y_test==============")
print(Y_test.ravel())
print("============Y_predict============")
print(Y_predict)
 
# #运行结果：
# R值(准确率): 0.970684039088
# 参数: [[ 1.3926311   0.17397478  0.65749877  0.8929026   0.36507062  1.36092964
#    0.91444624  0.63198866  0.75459326]]
# 截距: [-1.02717163]
# 稀疏化特征比率:0.00%
# =========sigmoid函数转化的值，即：概率p=========
# [[  6.61838068e-06   9.99993382e-01]
#  [  3.78575185e-05   9.99962142e-01]
#  [  2.44249065e-15   1.00000000e+00]
#  [  0.00000000e+00   1.00000000e+00]
#  [  1.52850624e-03   9.98471494e-01]
#  [  6.67061684e-05   9.99933294e-01]
#  [  6.75536843e-07   9.99999324e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  2.43117004e-05   9.99975688e-01]
#  [  6.13092842e-04   9.99386907e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  2.00330728e-06   9.99997997e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  3.78575185e-05   9.99962142e-01]
#  [  4.65824155e-08   9.99999953e-01]
#  [  5.47788703e-10   9.99999999e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  0.00000000e+00   1.00000000e+00]
#  [  0.00000000e+00   1.00000000e+00]
#  [  6.27260778e-07   9.99999373e-01]
#  [  3.78575185e-05   9.99962142e-01]
#  [  3.85098865e-06   9.99996149e-01]
#  [  1.80189197e-12   1.00000000e+00]
#  [  9.44640398e-05   9.99905536e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  0.00000000e+00   1.00000000e+00]
#  [  4.11688915e-06   9.99995883e-01]
#  [  1.85886872e-05   9.99981411e-01]
#  [  5.83016713e-06   9.99994170e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  1.52850624e-03   9.98471494e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  0.00000000e+00   1.00000000e+00]
#  [  1.51713085e-05   9.99984829e-01]
#  [  2.34685008e-05   9.99976531e-01]
#  [  1.51713085e-05   9.99984829e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  0.00000000e+00   1.00000000e+00]
#  [  2.34685008e-05   9.99976531e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  9.97563915e-07   9.99999002e-01]
#  [  1.70686321e-07   9.99999829e-01]
#  [  1.38382134e-04   9.99861618e-01]
#  [  1.36080718e-04   9.99863919e-01]
#  [  1.52850624e-03   9.98471494e-01]
#  [  1.68154251e-05   9.99983185e-01]
#  [  6.66097483e-04   9.99333903e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  9.77502258e-07   9.99999022e-01]
#  [  5.83016713e-06   9.99994170e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  4.09496721e-06   9.99995905e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  1.37819117e-06   9.99998622e-01]
#  [  6.27260778e-07   9.99999373e-01]
#  [  4.52734741e-07   9.99999547e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  8.88178420e-16   1.00000000e+00]
#  [  1.06976766e-08   9.99999989e-01]
#  [  0.00000000e+00   1.00000000e+00]
#  [  2.45780192e-04   9.99754220e-01]
#  [  3.92389040e-04   9.99607611e-01]
#  [  6.10681985e-05   9.99938932e-01]
#  [  9.44640398e-05   9.99905536e-01]
#  [  1.51713085e-05   9.99984829e-01]
#  [  2.45780192e-04   9.99754220e-01]
#  [  2.45780192e-04   9.99754220e-01]
#  [  1.51713085e-05   9.99984829e-01]
#  [  0.00000000e+00   1.00000000e+00]]
# =============Y_test==============
# [2 2 4 4 2 2 2 4 2 2 4 2 4 2 2 2 4 4 4 2 2 2 4 2 4 4 2 2 2 4 2 4 4 2 2 2 4
#  4 2 4 2 2 2 2 2 2 2 4 2 2 4 2 4 2 2 2 4 2 2 4 2 2 2 2 2 2 2 2 4]
# ============Y_predict============
# [2 2 4 4 2 2 2 4 2 2 4 2 4 2 2 2 4 4 4 2 2 2 4 2 4 4 2 2 2 4 2 4 4 2 2 2 4
#  4 2 4 2 2 2 2 2 2 2 4 2 2 4 2 4 2 2 2 4 4 2 4 2 2 2 2 2 2 2 2 4]
