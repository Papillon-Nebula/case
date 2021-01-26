import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model.coordinate_descent import ConvergenceWarning
import xgboost as xgb 
from sklearn.metrics import r2_score

from sklearn.datasets import load_boston

# warnings.filterwarnings('ignore')

# 1、加载数据
# names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT','MEDV']
data = load_boston()
Y = data.target
X = data.data

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=27)
print("训练集大小:{}".format(x_train.shape))
print("测试集大小:{}".format(x_test.shape))

# 使用XGBOOST的API
# a：数据转换
dtrain = xgb.DMatrix(data=x_train,label=y_train)
dtest = xgb.DMatrix(data=x_test)

# b：模型参数构建
params = {'max_depth':3, 'eta':1, 'silent':0, 'objective':'reg:linear'}
num_boost_round = 2

# c：模型训练
model = xgb.train(params=params, dtrain=dtrain,num_boost_round=num_boost_round)

# d：模型保存
# model.save_model('xgb.model')
print(model)

# 加载模型产生预测值
model2 = xgb.Booster()
model2.load_model('xgb.model')
print(model2)
print("训练集R2：{}".format(r2_score(y_train, model2.predict(dtrain))))
print("测试集R2：{}".format(r2_score(y_test, model2.predict(dtest))))