import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

#加载数据
data = load_boston()
Y = data.target
X = data.data

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=27)

'''
    max_depth : int
        Maximum tree depth for base learners.
    learning_rate : float
        Boosting learning rate (xgb's "eta")
    n_estimators : int
        Number of boosted trees to fit.
    silent : boolean
        Whether to print messages while running boosting.
    objective : string
        给定损失函数，默认为”binary:logistic” 给定损失函数，默认为”reg:linear”
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    reg_alpha : float (xgb's alpha)
        L1 regularization term on weights
    reg_lambda : float (xgb's lambda)
        L2 regularization term on weights
    base_score:
        The initial prediction score of all instances, global bias.
    random_state : int
'''

param_grid = {
    "max_depth":[5,6,7],
    "learning_rate":[0.1,1],
    "n_estimators":[10,35],
    "objective":["reg:linear"],
    "reg_alpha":[0.01,0.1],
    "reg_lambda":[0.3,0.5,1]
}

model= xgb.XGBRegressor()

#网格参数交叉验证
grid_search_cv = GridSearchCV(estimator = model, param_grid = param_grid)
grid_search_cv.fit(x_train,y_train)

y_hat = grid_search_cv.predict(x_test)
print("模型最优参数为:{}".format(grid_search_cv.best_params_))
print("在训练集上的R2={}".format(np.round(r2_score(y_train,grid_search_cv.predict(x_train)),3)))
print("在测试集上的R2={}".format(np.round(r2_score(y_test,y_hat),3)))

# 模型最优参数为:{'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 35, 'objective': 'reg:linear', 'reg_alpha': 0.01, 'reg_lambda': 0.5}
# 在训练集上的R2=0.98
# 在测试集上的R2=0.917

#输出特征的重要性列表
xgb.plot_importance(model,importance_type='cover')