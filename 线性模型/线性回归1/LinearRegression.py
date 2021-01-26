#线性回归——家庭用电预测（时间与功率之间的关系）
 
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
import time
 
#导入数据
path = "../datas/household_power_consumption_1000.txt"
data = pd.read_csv(path,sep=";")
 
#查看数据
print(data.head())     #查看头信息，默认前5行的数据
 
#iloc进行行列切片只能用数字下标，取出X的原始值（所有行与一、二列的表示时间的数据）
xdata = data.iloc[:,0:2]
# print(xdata)
 
y = data.iloc[:,2]      #取出Y的数据（功率）
#y = data["Global_active_power"]        #等价上面一句
# print(ydata)
 
#创建时间处理的函数
def time_format(x):
    #join方法取出的两列数据用空格合并成一列
    #用strptime方法将字符串形式的时间转换成时间元祖struct_time
    t = time.strptime(" ".join(x), "%d/%m/%Y %H:%M:%S")   #日月年时分秒的格式
    # 分别返回年月日时分秒并放入到一个元组中
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
 
#apply方法表示对xdata应用后面的转换形式
x = xdata.apply(lambda x:pd.Series(time_format(x)),axis=1)
print("======处理后的时间格式=======")
print(x.head())
 
#划分测试集和训练集，random_state是随机数发生器使用的种子
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
 
# print("========x_train=======")
# print(x_train)
# print("========x_text=======")
# print(x_test)
# print("========y_train=======")
# print(y_train)
# print("========y_text=======")
# print(y_test)
 
#对数据的训练集和测试集进行标准化
ss = StandardScaler()
#fit做运算，计算标准化需要的均值和方差；transform是进行转化
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
 
#建立线性模型
lr = LinearRegression()
lr.fit(x_train,y_train)     #训练
print("准确率:",lr.score(x_train,y_train))    #打印预测的决定系数,该值越接近于1越好
y_predict = lr.predict(x_test)  #预测
# print(lr.score(x_text,y_predict))
 
#模型效果判断
mse = np.average((y_predict-np.array(y_test))**2)
rmse = np.sqrt(mse)
print("均方误差平方和：",mse)
print("均方误差平方和的平方根：",rmse)
 
#模型的保存与持久化
from sklearn.externals import joblib
 
joblib.dump(ss,"data_ss.model")     #将标准化模型保存
joblib.dump(lr,"data_lr.model")     #将训练后的线性模型保存
 
joblib.load("data_ss.model")        #加载模型,会保存该model文件
joblib.load("data_lr.model")        #加载模型
 
#预测值和实际值画图比较
 
#解决中文问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
 
t = np.arange(len(x_test))
plt.figure(facecolor="w")       #创建画布，facecolor为背景色，w是白色（默认）
plt.plot(t,y_test,"r-",linewidth = 2,label = "真实值")
plt.plot(t,y_predict,"g-",linewidth = 2,label = "预测值")
plt.legend(loc = "upper right")    #显示图例，设置图例的位置
plt.title("线性回归预测时间和功率之间的关系",fontsize=20)
plt.grid(b=True)
plt.savefig("线性回归预测时间和功率之间的关系.png")     #保存图片
plt.show()
