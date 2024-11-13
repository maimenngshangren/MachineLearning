import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# K-近邻
# # 1. 构造数据
# x = [[1], [2], [10], [20]]
# y = [0, 0, 1, 1]
#
# # 2. 训练模型
# # 2.1 实例化一个估计器对象
# estimator = KNeighborsClassifier(n_neighbors=1)
#
# # 2.2 调用fit方法，进行训练
# estimator.fit(x,y)
#
# # 3 数据预测
# ret=estimator.predict([[0]])
# print(ret)
#
# ret1 = estimator.predict([[100]])
# print(ret1)


# *************鸢尾花种类预测
# 1 数据集获取
# 1.1 小数据集获取
# iris = load_iris()
# print(iris)

# 1.2 大数据集获取
# news = fetch_20newsgroups
# print(news)

# 2 数据集属性描述
# print('数据集的特征值:\n', iris.data)
# print('数据集的目标值:\n', iris['target'])
# print('数据集的特征值名字:\n', iris.feature_names)
# print('数据集的目标值名字:\n', iris.target_names)
# print('数据集的描述:\n', iris.DESCR)

# 3 数据可视化
# iris_d = pd.DataFrame(iris['data'], columns=['speal_length', 'speal_width', 'petal_length', 'petal_width'])
# iris_d['species'] = iris.target
#
#
# def iris_plot(data, col1, col2):
#     sns.lmplot(x=col1, y=col2, data=data, hue='species', fit_reg=False)
#     plt.title('鸢尾花数据显示')
#     plt.show()


# iris_plot(iris_d, 'speal_length', 'speal_width')
# iris_plot(iris_d, 'speal_width', 'petal_length')

# 4 数据划分

# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
# print('训练集的特征值是:\n', x_train)
# print('测试集的特征值是:\n', x_test)
# print('训练集的目标值是:\n', y_train)
# print('测试集的目标值是:\n', y_test)
#
# print('训练集的目标值的形状:\n', y_train.shape)
# print('测试集的目标值的形状:\n', y_test.shape)
#
# x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
# print('测试集的目标值是:\n', y_test)
# print('测试集的目标值是:\n', y_test1)
# print('测试集的目标值是:\n', y_test2)


# 数据处理练习
# def minmax_demo():
#     """
#     归一化演示
#     :return: None
#     """
#
#     data = pd.read_csv('./data/dating.txt')
#     print(data)
#
#     # 实例化
#     transfer = MinMaxScaler(feature_range=(3, 5))
#
#     # 进行转换
#     ret_data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
#     print('归一化之后的数据是:\n', ret_data)
#
#
# minmax_demo()
#
# def Stand_demo():
#     """
#     标准化演示
#     :return:None
#     """
#
#     data = pd.read_csv('./data/dating.txt')
#     print(data)
#
#     # 实例化
#     transfer = StandardScaler()
#
#     # 进行转换
#     ret_data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
#     print('归一化之后的数据是:\n', ret_data)
#     print('每一列的方差是:\n', transfer.var_)
#     print('每一列的平均值是:\n', transfer.mean_)


# 完整流程
# 1. 获取数据
iris = load_iris()

# 2. 数据基本处理
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# 3. 特征工程
# 实例化
transfer = StandardScaler()
# 转换
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4. KNN训练
# 实例化
estimator = KNeighborsClassifier()

# 模型选择与调优--网格收缩搜索和交叉验证
param_grid = {'n_neighbors': [1, 3, 5, 7]}
estimator = GridSearchCV(estimator, param_grid, cv=3)

# 训练
estimator.fit(x_train, y_train)

# 保存模型
joblib.dump(estimator, './data/test.pkl')

# 加载模型
estimator = joblib.load('./data/test.pkl')

# 5. 模型评估
# 预测值结果输出
y_pre = estimator.predict(x_test)
print('预测值是:\n', y_pre)
print('预测值和真实值对比结果是:\n', y_pre == y_test)
# 准确率计算
score = estimator.score(x_test, y_test)
print('准确率为:\n', score)

# 交叉验证和网格搜索的一些属性
print('在交叉验证中，得到的最好结果是:\n', estimator.best_score_)
print('在交叉验证中，得到的最好模型是:\n', estimator.estimator)
print('在交叉验证中，得到的模型结果是:\n', estimator.cv_results_)


# FaceBook案例
# # 1.获取数据集
# data = pd.read_csv('./data/train.csv')
# print(data.describe())
#
# # 2.基本数据处理
# # 2.1 缩⼩数据范围
# partial_data = data.query('x > 2.0 & x < 2.5 & y > 2.0 & y < 2.5')
# # 2.2 选择时间特征
# time = pd.to_datetime(partial_data['time'], unit='s')
# time = pd.DatetimeIndex(time)
# partial_data['hour'] = time.hour
# partial_data['day'] = time.day
# partial_data['weekday'] = time.weekday
# # print(partial_data.head())
# # 2.3 去掉签到较少的地⽅
# place_count = partial_data.groupby('place_id').count()
# place_count = place_count[place_count['row_id'] > 3]
# # print(partial_data.head())
# partial_data = partial_data[partial_data['place_id'].isin(place_count.index)]
# # print(partial_data.shape)
# # 2.4 确定特征值和⽬标值
# x = partial_data[['x', 'y', 'accuracy', 'hour', 'day', 'weekday']]
# y = partial_data['place_id']
# # 2.5 分割数据集
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.25)
#
# # 3.特征⼯程 -- 特征预处理(标准化)
# transfer = StandardScaler()
# x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
#
# # 4.机器学习 -- knn+cv
# # 实例化一个训练器
# estimator = KNeighborsClassifier()
#
# # 交叉验证、网格搜索实现
# param_grid = {'n_neighbors': [3, 5, 7, 9]}
# estimator = GridSearchCV(estimator, param_grid, cv=3, n_jobs=-1)   # n_jobs 用多少个CPU跑模型
#
# # 训练模型
# estimator.fit(x_train, y_train)
#
# # 5.模型评估
# score_ret = estimator.score(x_test, y_test)
# print(f'准确率为:+{score_ret}')
# y_pre = estimator.predict(x_test)
# print(f'预测值是:{y_pre}')
# print(f'最好的模型是:{estimator.best_estimator_}')
# print(f'最好的结果是:{estimator.best_score_}')
# print(f'所有的结果是:{estimator.cv_results_}')

# 留一法
# data = [1, 2, 3, 4]
# loo = LeaveOneOut
#
# for train, test in loo.split(data):
#     print('%s,%s' % (train, test))

# 交叉验证法  ————KFold, StratifiedKFold
X = np.array([
    [1, 2, 3, 4],
    [11, 12, 13, 14],
    [21, 22, 23, 24],
    [31, 32, 33, 34],
    [41, 42, 43, 44],
    [51, 52, 53, 54],
    [61, 62, 63, 64],
    [71, 72, 73, 74]
])

y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
folder = KFold(n_splits=4, shuffle=False)
sfolder = StratifiedKFold(n_splits=4, shuffle=False)

# KFold
print('KFold')
for train, test in folder.split(X, y):
    print(f'train:{test},tset:{test}')

# StratifiedKFold
print('StratifiedKFold')
for train, test in sfolder.split(X, y):
    print(f'train:{test},tset:{test}')