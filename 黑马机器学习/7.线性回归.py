import pandas
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

import joblib

# x = [[80, 86],
# [82, 80],
# [85, 78],
# [90, 90],
# [86, 82],
# [82, 90],
# [78, 80],
# [92, 94]]
# y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
#
# # 实例化一个估计器
# estimator = LinearRegression()
# # 进行训练
# estimator.fit(x,y)
#
# print(f'线性回归的系数是:{estimator.coef_}')
# print(f'输出的结果是:{estimator.predict([[100, 88]])}')


# 正规方程
# 1 获取数据
# boston = load_boston

# 2 数据基本处理
# # 分割数据
# x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
#
# # 3 特征工程--标准化
# transfer = StandardScaler()
# x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
#
# # 4 机器学习--线性回归
# estimator = LinearRegression()
# estimator.fit(x_train, y_train)
# print(f'这个模型的偏置是:{estimator.intercept_}')
# print(f'这个模型的系数是:{estimator.coef_}')
#
# # 5 模型评估
# y_pre = estimator.predict(x_test)
# print(f'预测结果是:{y_pre}')
#
# ret = mean_squared_error(y_pre, y_test)
# print(f'均方误差是:{ret}')


# # 梯度下降法
# # 1 获取数据
# boston = load_boston
#
# # 2 数据基本处理
# # 分割数据
# x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
#
# # 3 特征工程--标准化
# transfer = StandardScaler()
# x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
#
# # 4 机器学习--梯度下降
# # estimator = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.001)
# estimator = SGDRegressor(max_iter=1000)
# estimator.fit(x_train, y_train)
#
# # 5 模型评估
# y_pre = estimator.predict(x_test)
# print(f'预测结果是:{y_pre}')
#
# ret = mean_squared_error(y_pre, y_test)
# print(f'均方误差是:{ret}')


# # 岭回归
# # 1 获取数据
# boston = load_boston
#
# # 2 数据基本处理
# # 分割数据
# x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
#
# # 3 特征工程--标准化
# transfer = StandardScaler()
# x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
#
# # 4 机器学习--梯度下降
# estimator = Ridge(alpha=1.0)
# estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
# estimator.fit(x_train, y_train)
#
# # 5 模型评估
# y_pre = estimator.predict(x_test)
# print(f'预测结果是:{y_pre}')
#
# ret = mean_squared_error(y_pre, y_test)
# print(f'均方误差是:{ret}')


# 案例
# # 1 获取数据
# names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
#          'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
#          'Normal Nucleoli', 'Mitoses', 'Class']
# data = pandas.read_csv(
#     "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
#     names=names)
#
# # 2 数据处理
# data = data.replace(to_replace='?', value=np.NaN)
# data = data.dropna()
# x = data.iloc[:, 1:10]
# y = data['Class']
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
#
# # 3 特征工程
# transfer = StandardScaler()
# x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
#
# # 4 机器学习--逻辑回归
# estimator = LogisticRegression()
# estimator.fit(x_train, y_train)
#
# # 5 模型评估
# y_predict = estimator.predict(x_test)
# print(y_predict)
# ret = estimator.score(x_test, y_test)
# print(f'准确率为:{ret}')
# # 5.1 分类评估
# ret_1 = classification_report(y_test, y_predict, labels=(2, 4), target_names=['良性', '恶性'])
# print(ret_1)
#
# # 5.2 ROC/AUC指标
# y_test = np.where(y_test > 3, 1, 0)
# print('AUC指标:', roc_auc_score(y_test, y_predict))


# 类别不平衡数据
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_repeated=0, n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94], random_state=0)
print(X.shape)
print(Counter(y))

# 数据可视化
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# 随机过采样
ros = RandomOverSampler(random_state=0)
x_resampled, y_rasampled = ros.fit_resample(X, y)
print(Counter(y_rasampled))

plt.scatter(x_resampled[:, 0], x_resampled[:, 1], c=y_rasampled)
plt.show()

# SMOTE方法
ste = SMOTE()
x_smote, y_smote = ste.fit_resample(X, y)
print(Counter(y_smote))

plt.scatter(x_smote[:,0],x_smote[:,1],c=y_smote)
plt.show()

# 降采样
rus = RandomUnderSampler(random_state=0)
x_undersample, y_undersample = rus.fit_resample(X,y)
print(Counter(y_undersample))


