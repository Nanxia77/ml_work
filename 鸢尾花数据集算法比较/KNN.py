# 文件功能：knn实现鸢尾花数据集分类
from sklearn import datasets  # 引入sklearn包含的众多数据集
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from sklearn.neighbors import KNeighborsClassifier  # 利用knn方式训练数据
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
# 【1】引入训练数据
iris = load_iris()
X = X = iris.data[:, :2]   #获取花卉两列数据集
Y = iris.target
#1knn
knn = KNeighborsClassifier()  # 引入训练方法
knn.fit(X, Y)  # 进行填充测试数据进行训练
#2逻辑回归模型
lr = LogisticRegression()
lr.fit(X,Y)
# 【3】训练svm分类器
clf = svm.SVC()
clf.fit(X, Y)
#4随机森林模型
RF = RandomForestClassifier()
RF.fit(X, Y)
#5决策树
dt_model = DecisionTreeClassifier()  # 所有参数均置为默认状态
dt_model.fit(X, Y)
### 效果评估
svm_score1 = accuracy_score(Y, clf.predict(X))
lr_score1 = accuracy_score(Y, lr.predict(X))
RF_score1 = accuracy_score(Y, RF.predict(X))
knn_score1 =accuracy_score(Y, knn.predict(X))
dt_score1 =accuracy_score(Y, dt_model.predict(X))
## 计算模型的准确率/精度
print (lr.score(X, Y))
print ('逻辑回归训练集准确率：', accuracy_score(Y, lr.predict(X)))
print (RF.score(X, Y))
print ('随机森林回归训练集准确率：', accuracy_score(Y, RF.predict(X)))
print (knn.score(X, Y))
print ('k临近算法回归训练集准确率：', accuracy_score(Y, knn.predict(X)))
print (dt_model.score(X, Y))
print ('决策树算法回归训练集准确率：', accuracy_score(Y, dt_model.predict(X)))
## 计算决策函数的结构值以及预测值(decision_function计算的是样本x到各个分割平面的距离<也就是决策函数的值>)
print ('decision_function:\n', clf.decision_function(X))
print ('\npredict:\n', clf.predict(X))


## 画图
x_tmp = [0,1,2,3]
y_score1 = [ lr_score1, RF_score1, knn_score1,dt_score1]


plt.figure(facecolor='w')
plt.plot(x_tmp, y_score1, 'r-', lw=2, label=u'accuracy')
plt.xlim(0, 3)
plt.ylim(np.min((np.min(y_score1), np.min(y_score1)))*0.9, np.max((np.max(y_score1), np.max(y_score1)))*1.1)
plt.legend(loc = 'lower right')
plt.title(u'Comparison of accuracy of different classifiers for Iris data', fontsize=12)
plt.xticks(x_tmp, [u'Logistic', u'RF', u'KNN',u'DT'], rotation=0)
plt.grid(b=True)
plt.show()


#meshgrid函数生成两个网格矩阵
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#pcolormesh函数将xx,yy两个网格矩阵和对应的预测结果Z绘制在图片上
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
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
plt.title('KNeighborsClassifier')
plt.show()


