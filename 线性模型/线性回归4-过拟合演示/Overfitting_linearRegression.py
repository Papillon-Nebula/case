#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu

#过拟合样例代码
import sys
sys.path.append('./')
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.exceptions import ConvergenceWarning

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

#解决中文问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

np.random.seed(100)  #seed() 设置生成随机数用的整数起始值
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x**3 + x**2 - 14 * x - 7 + np.random.randn(N)

x.shape = -1, 1  #转完成一列
y.shape = -1, 1

models = [
    Pipeline([('Poly', PolynomialFeatures()),
              ('Linear', LinearRegression(fit_intercept=False))]),
    Pipeline([('Poly', PolynomialFeatures()),
              ('Linear',
               RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
    Pipeline([('Poly', PolynomialFeatures()),
              ('Linear',
               LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
    Pipeline([('Poly', PolynomialFeatures()),
              ('Linear',
               ElasticNetCV(alphas=np.logspace(-3, 2, 50),
                            l1_ratio=[.1, .5, .7, .9, .95, 1],
                            fit_intercept=False))])
]

plt.figure(facecolor='w')  #创建画布
degree = np.arange(1, N, 4)  #阶数（一阶，五阶，九阶）
dm = degree.size
colors = []
for c in np.linspace(16711680, 255, dm):
    c = c.astype(int)
    colors.append('#%06x' % c)

model = models[0]

for i, d in enumerate(degree):
    plt.subplot(int(np.ceil(dm / 2.0)), 2, i + 1)
    plt.plot(x, y, 'ro', ms=10, zorder=N)

    model.set_params(Poly__degree=d)
    model.fit(x, y.ravel())

    lin = model.get_params('Linear')['Linear']
    output = u'%d阶，系数为' % d
    print(output, lin.coef_.ravel())

    x_hat = np.linspace(x.min(), x.max(), num=100)
    x_hat.shape = -1, 1
    y_hat = model.predict(x_hat)
    s = model.score(x, y)

    z = N - 1 if (d == 2) else 0
    label = u"%d阶，准确率为：%.3f" % (d, s)
    plt.plot(x_hat,
             y_hat,
             color=colors[i],
             lw=2,
             alpha=0.75,
             label=label,
             zorder=z)

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'线性回归过拟合显示', fontsize=22)
plt.savefig('线性回归过拟合显示.png')
plt.show()

# #运行结果：
# 1阶，系数为 [-44.14102611  40.05964256]
# 5阶，系数为 [ -5.60899679 -14.80109301   0.75014858   2.11170671  -0.07724668   0.00566633]
# 9阶，系数为 [-2465.5996245   6108.67810881 -5112.02743837   974.75680049  1078.90344647  -829.50835134   266.13413535   -45.7177359      4.11585669    -0.15281174]
