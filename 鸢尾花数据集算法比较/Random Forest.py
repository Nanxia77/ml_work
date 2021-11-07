# 文件功能：随机森林分类鸢尾花数据集
"""
随机森林主要应用于回归和分类两种场景，侧重于分类。随机森林是指利用多棵树对样本数据进行训练、分类并预测的一种方法。
它在对数据进行分类的同时，还可以给出各个变量的重要性评分，评估各个变量在分类中所起的作用。
"""
"""
随机森林的构建：
1.首先利用bootstrap方法有放回地从原始训练集中随机抽取n个样本，并构建n个决策树；
2.然后假设在训练样本数据中有m个特征，那么每次分裂时选择最好的特征进行分裂，每棵树都一直这样分裂下去，直到该节点
3.的所有训练样例都属于同一类；接着让每棵决策树在不做任何修剪的前提下最大限度地生长；
4.最后将生成的多棵分类树组成随机森林，用随机森林分类器对新的数据进行分类与回归。对于分类问题，按多棵树分类器投票决定最终分类结果；对于回归问题，则由多棵树预测值的均值决定最终预测结果
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

RF = RandomForestClassifier()
iris = load_iris()
X = X = iris.data[:, :2]   #获取花卉两列数据集
Y = iris.target
RF.fit(X, Y)
#meshgrid函数生成两个网格矩阵
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#pcolormesh函数将xx,yy两个网格矩阵和对应的预测结果Z绘制在图片上
Z = RF.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(8,6))
plt.pcolormesh(xx, yy, Z, shading='auto',cmap=plt.cm.Paired)

#绘制散点图
plt.scatter(X[:50,0], X[:50,1], color='red',marker='o', label='setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='versicolor')
plt.scatter(X[100:,0], X[100:,1], color='green', marker='s', label='Virginica')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=2)
plt.title('RandomForestClassifier')
plt.show()