#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu
 
#鸢尾花数据分类——决策树
# import sys
# sys.path.append('./')
from sklearn import tree        #决策树
from sklearn.tree import DecisionTreeClassifier     #决策分类树
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV       #网格搜索交叉验证
from sklearn.pipeline import Pipeline                   #管道
from sklearn.preprocessing import MinMaxScaler      #数据归一化
from sklearn.feature_selection import SelectKBest   #特征选择
from sklearn.feature_selection import chi2          #卡方统计量
from sklearn.decomposition import PCA       #主成分分析
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
 
#解决中文显示问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
 
#导入数据
path = "Datas\iris.data"
data = pd.read_csv(path,header=None)
 
iris_feature_E = "sepal length","sepal width","petal length","petal width"
iris_feature_C = u"花萼长度",u"花萼宽度",u"花瓣长度",u"花瓣宽度"
iris_class = "Iris-setosa","Iris-versicolor","Iris-virginica"
 
#数据分割
x = data[np.arange(0,4)]    #获取x变量
#x = data[list(range(4))]   #与上面一句等价
#print(x.head())
y = pd.Categorical(data[4]).codes  #Categorical:编码包含大量重复文本的数据，codes把数据y转换成分类型的0，1,2
print("样本总数:%d;特征属性数目:%d" %x.shape)
print(y)
 
#划分训练集与测试集
x_train1, x_test1, y_train1, y_test1 = train_test_split(x,y,test_size=0.2,random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print("训练数据集样本总数:%d;测试数据集样本总数:%d" %(x_train.shape[0],x_test.shape[0]))
 
#对数据集进行标准化
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train,y_train)
x_test = ss.transform(x_test)
print("原始数据各个特征的调整最小值:",ss.min_)
print("原始数据各个特征的缩放数据值:",ss.scale_)
 
#特征选择：从已有的特征属性中选择出影响目标最大的特征属性
#常用方法：{分类：F统计量、卡方系数、互信息mutual_info_classif
#           连续：皮尔逊相关系数、F统计量、互信息mutual_info_classif}
#SelectKBest(卡方系数)
ch2 = SelectKBest(chi2,k=3) #当前案例中，用SelectKBest方法从四个原始特征属性中选择出最能影响目标的3个特征属性
                            # k 默认为10，指定后会返回想要的特征个数
x_train = ch2.fit_transform(x_train,y_train)    #训练并转换
x_test = ch2.transform(x_test)      #转换
select_name_index = ch2.get_support(indices=True)
print("对类别判别影响最大的三个特征属性分别是:",ch2.get_support(indices=False))
print(select_name_index)
 
#降维：对于数据而言，如果特征属性比较多，在构建过程中会比较复杂，
# 这时将多维（高维）降到低维空间中
#常用的降维方法：PCA  主成分分析（无监督）；人脸识别通常先做一次PCA
# LDA  线性判别分析（有监督），类内方差最小
 
pca = PCA(n_components=2)   #构建一个PCA对象，设置最终维度为2维
#这里为了后边画图方便，将数据维度设置为 2，一般用默认不设置就可以
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
 
#模型构建
model = DecisionTreeClassifier(criterion="entropy",random_state=0)
#模型训练
model.fit(x_train,y_train)
#模型预测
y_test_hat = model.predict(x_test)
 
#利用数据可视化软件Graphviz打印出决策树
#from sklearn.externals.six import StringIO
#with open("iris.dot") as f:
    #f = tree.export_graphviz(model,out_file=f)
 
print("Score:",model.score(x_test,y_test))
print("Classes:",model.classes_)
 
N = 100
x1_min = np.min((x_train.T[0].min(),x_test.T[0].min()))
x1_max = np.max((x_train.T[0].max(),x_test.T[0].max()))
x2_min = np.min((x_train.T[1].min(),x_test.T[1].min()))
x2_max = np.max((x_train.T[1].max(),x_test.T[1].max()))
 
t1 = np.linspace(x1_min,x1_max,N)
t2 = np.linspace(x2_min,x2_max,N)
x1,x2 = np.meshgrid(t1,t2)  #生成网格采样点
x_show = np.dstack((x1.flat,x2.flat))[0]
y_show_hat = model.predict(x_show)
y_show_hat = y_show_hat.reshape(x1.shape)
print(y_show_hat.shape)
print(y_show_hat[0])
 
#画图
plt_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor="w")
plt.pcolormesh(x1,x2,y_show_hat,cmap=plt_light)
plt.scatter(x_test.T[0],x_test.T[1],c=y_test.ravel(),edgecolors="k",
            s=150,zorder=10,cmap=plt_dark,marker="*")       #测试数据
plt.scatter(x_train.T[0],x_train.T[1],c=y_train.ravel(),edgecolors="k",
            s=40,cmap=plt_dark)     #全部数据
plt.xlabel(u"特征属性1",fontsize=15)
plt.ylabel(u"特征属性2",fontsize=15)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.grid(True)
plt.title(u"鸢尾花数据的决策树分类",fontsize=18)
plt.savefig("ML\case\鸢尾花数据的决策树分类.png")
plt.show()
 
#参数优化
pipe = Pipeline([
            ('mms', MinMaxScaler()),
            ('skb', SelectKBest(chi2)),
            ('pca', PCA()),
            ('decision', DecisionTreeClassifier())
        ])
 
# 参数
parameters = {
    "skb__k": [1,2,3,4],
    "pca__n_components": [0.5,1],
    "decision__criterion": ["gini", "entropy"],
    "decision__max_depth": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
}
 
x_train2, x_test2, y_train2, y_test2 = x_train1, x_test1, y_train1, y_test1
 
gscv = GridSearchCV(pipe, param_grid=parameters)
 
gscv.fit(x_train2, y_train2)
 
print("最优参数列表:",gscv.best_params_)
print ("score值：",gscv.best_score_)
 
y_test_hat2 = gscv.predict(x_test2)
 
mms_best = MinMaxScaler()
skb_best = SelectKBest(chi2,k=2)
pca_best = PCA(n_components=0.5)
decision3 = DecisionTreeClassifier(criterion="gini",max_depth=2)
x_train3, x_test3, y_train3, y_test3 = x_train1, x_test1, y_train1, y_test1
x_train3 = pca_best.fit_transform(skb_best.fit_transform(mms_best.fit_transform(x_train3,y_train3),y_train3))
x_test3 = pca_best.transform(skb_best.transform(mms_best.transform(x_test3)))
decision3.fit(x_train3,y_train3)
print("正确率:",decision3.score(x_test3,y_test3))
 
x_train4, x_test4, y_train4, y_test4 = train_test_split(x.iloc[:, :2], y, train_size=0.7, random_state=14)
 
depths = np.arange(1, 15)
err_list = []
for d in depths:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=d)
    clf.fit(x_train4, y_train4)
 
    score = clf.score(x_test4, y_test4)
    err = 1 - score
    err_list.append(err)
    print("%d深度，正确率%.5f" % (d, score))
 
 
## 画图
plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel(u'决策树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.grid(True)
plt.title(u'决策树层次太多导致的拟合问题(欠拟合和过拟合)', fontsize=18)
plt.savefig("ML\case\决策树层次太多导致的拟合问题(欠拟合和过拟合).png")
plt.show()
 
# #运行结果：
# 样本总数:150;特征属性数目:4
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# 训练数据集样本总数:120;测试数据集样本总数:30
# 原始数据各个特征的调整最小值: [-1.19444444 -0.83333333 -0.18965517 -0.04166667]
# 原始数据各个特征的缩放数据值: [ 0.27777778  0.41666667  0.17241379  0.41666667]
# 对类别判别影响最大的三个特征属性分别是: [ True False  True  True]
# [0 2 3]
# Score: 0.966666666667
# Classes: [0 1 2]
# (100, 100)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
# 最优参数列表: {'skb__k': 2, 'decision__max_depth': 2, 'pca__n_components': 0.5, 'decision__criterion': 'gini'}
# score值： 0.933333333333
# 正确率: 1.0
# 1深度，正确率0.55556
# 2深度，正确率0.73333
# 3深度，正确率0.77778
# 4深度，正确率0.73333
# 5深度，正确率0.68889
# 6深度，正确率0.68889
# 7深度，正确率0.68889
# 8深度，正确率0.66667
# 9深度，正确率0.66667
# 10深度，正确率0.66667
# 11深度，正确率0.66667
# 12深度，正确率0.66667
# 13深度，正确率0.66667
# 14深度，正确率0.66667
