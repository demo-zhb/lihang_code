import numpy as np
import matplotlib.pyplot as plt


def createdata():
    samples = np.array([[3, 3], [4, 3], [1, 1]])
    lables = [1, 1, -1]
    return samples, lables


class Perceptron():
    def __init__(self, x, y, a=1):
        self.x = x
        self.y = y
        # 初始化w0，b0
        self.w = np.zeros(x[0].shape)
        self.b = 0
        # 学习率设为1
        self.lr = 1
        self.numsamples = self.x.shape[0]  # 实例个数N
        # Gram矩阵
        self.G = np.zeros((self.numsamples, self.numsamples))
        for i in range(self.numsamples):
            for j in range(self.numsamples):
                self.G[i][j] = np.dot(self.x[i], self.x[j])
        self.alpha = np.zeros(self.numsamples)

    def sign(self, i):
        y = self.b
        for j in range(self.numsamples):
            y += self.alpha[j]*self.y[j]*self.G[i][j]
        return int(y)

    def update(self, i):
        self.alpha[i] += self.lr
        self.b += self.y[i]

    def cal_w(self):
        for i in range(self.numsamples):
            self.w += self.alpha[i]*self.y[i]*self.x[i]

    def train(self):
        Flag_Find = False
        iterations = 0          # 表示迭代次数
        while not Flag_Find:
            # 假设已经找到可分离超平面S: dot(w,x)+b=0
            Flag_Find = True
            # 验证假设
            for i in range(self.numsamples):
                tmpY = self.sign(i)
                if tmpY * self.y[i] <= 0:
                    self.update(i)
                    print('误分类点：(',i, self.x[i], '),\n此时的b为：',
                          self.b,'\n', self.alpha)
                    iterations += 1
                    Flag_Find = False
        self.cal_w()
        print('迭代次数',iterations)
        print('alpha', self.alpha)
        print('最终训练得到的w和b为：', self.w, self.b)
        return self.w, self.b

# 画图描绘
class Picture():
    def __init__(self, data, w, b):
        self.data = data
        self.w = w
        self.b = b
        plt.figure(1)
        plt.title('Perceptron Learning Algorithm(PLA)',size=14)
        plt.xlabel('x0_axis', size=14)
        plt.ylabel('x1_axis', size=14)

        xData = np.linspace(0, 5, 100)
        yData = self.expression(xData)

        plt.plot(xData, yData, color='r', label='Sample data')
        plt.scatter(data[0][0], data[0][1], s=50, marker='o', c='b')
        plt.scatter(data[1][0], data[1][1], s=50, marker='o', c='b')
        plt.scatter(data[2][0], data[2][1], s=50, marker='x', c='g')
        plt.savefig('dual_form_result.jpg')
    def expression(self, xData):
        y = (-self.b-self.w[0]*xData)/self.w[1]
        return y

    def Show(self):
        plt.show()


if __name__ == '__main__':
    samples, labels = createdata()
    myperceptron = Perceptron(x=samples, y=labels)
    weights, bias = myperceptron.train()
    Picture = Picture(samples, weights, bias)
    Picture.Show()





