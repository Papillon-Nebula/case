import sys
sys.path.append('./')
# 引入所需要的全部包
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time

# 通用机器学习编写流程
# 一、构造（加载）数据
# 二、数据清洗
# 三、根据需求和原始模型从最原始的特征属性中获取具体的特征属性矩阵X和目标属性矩阵Y
# 四、数据分割（将数据分割为训练数据和测试数据）
# train_test_split()
# 五、特征工程的操作
# 六、模型对象的构建
# 七、模型训练
# 八、模型效果的评估
# 九、模型保存/模型持久化
"""
方式一： 直接保存模型预测结果
方式二： 将模型持久化为磁盘文件
方式三： 将模型参数保存到数据库中，仅适合可以获取参数的模型，eg：线性回归
"""

# 一、构造（加载）数据
# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据
path = 'datas\household_power_consumption.txt'  ## 全部数据
path = 'datas\household_power_consumption_200.txt'  # 200行数据
path = 'datas\household_power_consumption_1000.txt'  # 1000行数据
df = pd.read_csv(path, sep=';',
                 low_memory=False)  # 没有混合类型的时候可以通过low_memory=False调用更多内存，加快效率

# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
names2 = df.columns
names = [
    'Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]

# print(df.head())  # 获取前五行数据查看
# print(df.describe())
# print(df.info())  # 查看格式信息

# 二、数据清洗
# inplace: 当设置为True的时候， 表示对原始的DataFrame做修改； 默认为False
# # 异常数据处理(异常数据过滤)
new_df = df.replace('?', np.nan)  # 替换非法字符为 np.nan
# DataFrame： axis=0 表示对行做处理， axis=1 表示对列作处理
# how： 可选参数 any 和 all ， any 表示只要有任意一个为nan，那么进行数据删除
# 功能： 只要有任意一个样本中的任意数据特征属性为 nan 的形式， 就将该样本删除
datas = new_df.dropna(axis=0, how='any')  # 只要有数据为空，就进行删除操作

# print(datas.describe())
# print(datas.describe().T)  # 观察数据的多种统计指标（只能看数值类型）


## 创建一个时间字符串格式化字符串
def date_format(dt):
    # dt显示是一个series/tuple： dt[0]是date，dt[1]是time
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

# 三、需求：构建时间和功率之间的映射关系，可以认为：特征属性为时间，目标属性为功率值(Linear)
# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]].values

# print(X.head())

# 四、数据分割（将数据分割为训练数据和测试数据）
# 对数据集进行测试集合训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=0)

# print(X_train.shape)

# 五、模型对象的构建
algo = LinearRegression(fit_intercept=False)

# 六、模型训练
# 在sklearn中，fit API的功能就是基于给定的训练数据进行模型训练，也就是找出输入数据X与输出数据之间的映射关系
# 在sklearn中，predict API的功能就是使用训练好的模型对输入的特征属性X做一个转换预测，得到输出值Y，该API仅在算法模型对象上存在。
algo.fit(X_train, Y_train)
# 七、模型效果的评估
# 模型校验
# y_predict = lr.predict(X_test)  ## 预测结果

# 八、模型效果评估
# predict_y = np.mat(X_test) * theta
predict_y = algo.predict(X_test)
y_test = np.mat(Y_test).reshape((-1, 1))
j_theta = np.mean(np.power(predict_y - Y_test, 2))
print(j_theta)
# print("训练集上的R2（准确率）:", lr.score(X_test, Y_test))
# print("测试集上的R2（准确率）:", lr.score(X_train, Y_train))
# mse = np.average((y_predict - Y_test) ** 2)   # 求均值 average
# rmse = np.sqrt(mse)
# print("rmse:", rmse)

# # 输出模型训练得到的相关参数
# print("模型的系数（Θ）：", end="")
# print(lr.coef_)
# print("模型的截距项：", end="")
# print(lr.intercept_)

# # 八、模型保存/模型持久化
# # 在机器学习部署的时候，实际上其中一种方式就是将模型进行输出； 另一种方式就是直接将预测结果输出
# # 模型输出一般是将模型输出到磁盘文件
# from sklearn.externals import joblib

# # 保存模型要求给定的文件所在的文件夹必须存在
# joblib.dump(ss, "ML/案例/result/data_ss.model")  # 将标准化模型保存
# joblib.dump(lr, "ML/案例/result/data_lr.model")  # 将模型保存

# # 九、加载模型
# ss3 = joblib.load("ML/案例/result/data_ss.model")  # 加载模型
# lr3 = joblib.load("ML/案例/result/data_lr.model")  # 加载模型

# # 使用加载的模型进行预测
# data1 = [[2006, 12, 17, 12, 25, 0]]
# data1 = ss3.transform(data1)
# print(data1)
# lr3.predict(data1)

# ## 预测值和实际值画图比较
# t = np.arange(len(X_test))
# plt.figure(facecolor='w')  # 建一个画布，facecolor 是背景色
# plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
# plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
# plt.legend(loc='lower right')  # 显示图例， 设置图例的位置
# plt.title(u"线性回归预测时间和功率之间的关系", fontsize=20)
# plt.grid(b=True)  #网格
# plt.show()

# ## 时间和电压之间的关系(Linear-多项式)
# models = [
#     Pipeline([
#         ('Poly', PolynomialFeatures(degree=3)),  # 给定进行多项式扩展操作
#         ('Linear', LinearRegression(fit_intercept=False))
#     ])
# ]
# model = models[0]
# # 获取x和y变量, 并将时间转换为数值型连续变量
# X = datas[names[0:2]]
# X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
# Y = datas[names[4]]

# # 对数据集进行测试集合训练集划分
# X_train, X_test, Y_train, Y_test = train_test_split(X,
#                                                     Y,
#                                                     test_size=0.2,
#                                                     random_state=0)

# # 数据标准化
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)  # 训练并转换
# X_test = ss.transform(X_test)  ## 直接使用在模型构建数据上进行一个数据标准化操作

# # 模型训练
# t = np.arange(len(X_test))
# N = 5
# d_pool = np.arange(1, N, 1)  # 阶
# m = d_pool.size
# clrs = []  # 颜色
# for c in np.linspace(16711680, 255, m):
#     clrs.append('#%06x' % int(c))
# line_width = 3

# plt.figure(figsize=(12, 6), facecolor='w')  #创建一个绘图窗口，设置大小，设置颜色
# for i, d in enumerate(d_pool):
#     plt.subplot(N - 1, 1, i + 1)
#     plt.plot(t, Y_test, 'r-', label=u'真实值', ms=10, zorder=N)
#     model.set_params(Poly__degree=d)  ## 设置多项式的阶乘
#     model.fit(X_train, Y_train)
#     lin = model.get_params('Linear')['Linear']
#     output = u'%d阶，系数为：' % d
#     if hasattr(lin, 'alpha_'):
#         idx = output.find(u'系数')
#         output = output[:idx] + (u'alpha=%.6f, ' % lin.alpha_) + output[idx:]
#     if hasattr(lin, 'l1_ratio_'):
#         idx = output.find(u'系数')
#         output = output[:idx] + (u'l1_ratio=%.6f, ' %
#                                  lin.l1_ratio_) + output[idx:]
#     print(output, lin.coef_.ravel())

#     y_hat = model.predict(X_test)
#     s = model.score(X_test, Y_test)

#     z = N - 1 if (d == 2) else 0
#     label = u'%d阶, 准确率=%.3f' % (d, s)
#     plt.plot(t,
#              y_hat,
#              color=clrs[i],
#              lw=line_width,
#              alpha=0.75,
#              label=label,
#              zorder=z)
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.ylabel(u'%d阶结果' % d, fontsize=12)

# ## 预测值和实际值画图比较
# plt.suptitle(u"线性回归预测时间和功率之间的多项式关系", fontsize=20)
# plt.grid(b=True)
# plt.show()
