import pandas as pd
import numpy as np
from  sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])
print(data)
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])

# 数据线性可分，二分类数据
# 此处为一元一次线性方程

class Model():
    def __init__(self):
        self.w = np.ones(len(data[0])-1, dtype=np.float32)
        self.b = 0
        self.lr = 0.1

    def sign(self, x, w, b):
        y = np.dot(w, x) + b
        return y

    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.lr * np.dot(y, X)
                    self.b = self.b + self.lr * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return self.w, self.b

    def score(self):
        pass


class Picture():
    def __init__(self, data, w, b):
        self.data = data
        self.w = w
        self.b = b
        plt.figure(1)
        plt.title('Perceptron Learning Algorithm',size=14)
        plt.xlabel('x0-axis', size=14)
        plt.ylabel('x1-axis', size=14)

        xData = np.linspace(4, 7, 10)
        yData = self.expression(xData)

        plt.plot(xData, yData, color='g')
        plt.scatter(data[:50, 0], data[:50, 1], s=50, label= '0')
        plt.scatter(data[50:100, 0], data[50:100, 1], s=50, label = '1')
        plt.legend()
        # plt.savefig('2d.png', dpi=75)

    def expression(self, xData):
        y = (-self.b-self.w[0]*xData)/self.w[1]
        return y

    def Show(self):
        plt.show()


perceptron = Model()
weights, bias = perceptron.fit(X, y)
Picture = Picture(data, weights, bias)
Picture.Show()
