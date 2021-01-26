import numpy as np
import matplotlib.pyplot as plt

# 0.8 = 10a + b + e1
# 1.0 = 15a + b + e2
# 1.8 = 20a + b + e3
# 2.0 = 30a + b + e4
# 3.2 = 40a + b + e5
# 3.0 = 50a + b + e6
# 3.1 = 60a + b + e7
# 3.5 = 70a + b + e8

# flag 判断是否加截距项(为True加，flase不加)
flag = True
# 一、构造数据
X1 = np.array([[10, 1], [15, 1], [20, 1], [30, 1], [50, 2], [60, 1], [60, 2],
               [70, 2]]).reshape((-1, 2))
Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))

if flag:
    # 添加一个截距项对应的X值
    # X = np.hstack((np.ones_like(X1), X1))
    X = np.hstack((np.ones(shape=(X1.shape[0], 1)), X1))
else:
    X = X1  # 不加入截距项
print(X)

# 二、为了求解比较方便，将numpy的‘numpy.ndarray’的数据类型转换为矩阵的形式。
X = np.mat(X)  # mat()是数据类型转换为矩阵的方法
Y = np.mat(Y)  # mat()是数据类型转换为矩阵的方法
# print(X)
print(Y)

# 三、根据解析式的公式求解theta的值
theta = (X.T * X).I * X.T * Y
print(theta)

# 四、根据求解出的theta求出预测值
predict_y = X * theta

# 五、画图可视化(这里的图像只能显示二维，如果需要显示三维的图像需要修改)
# plt.plot(X1, Y, 'bo', label=u'真实值')
# plt.plot(X1, predict_y, 'r-o', label='预测值')
# plt.legend(loc='lower rigt')
# plt.show()

# 基于训练好的模型参数对一个位置的样本进行预测
x = np.mat(np.array([[1.0, 55.0, 2.0]]))
pred_y = x * theta
print('当面积为55平且房间数为2 时，预测价格为：{}'.format(pred_y))