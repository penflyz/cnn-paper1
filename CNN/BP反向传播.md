

# BP反向传播

## 1 内容简介

主要涉及的思想是有均方误差（MSE）、链式法则和梯度下降等；

分为两个方面讲：1. 输出层-->隐含层 2. 隐含层-->隐含层

已知神经网络结构（**激活函数为sigmoid**）：

<img src="https://images2015.cnblogs.com/blog/853467/201606/853467-20160630142019140-402363317.png" alt="img" style="zoom: 33%;" />

## 2 前向传播

省略

## 3 反向传播

### 3.1 计算总误差(MSE)

这个是单个训练参数的误差，其中$y_k$是期望的输出，$o_k$是预测的输出。

这里加上$\frac{1}{2}$的原因是为了**后面的计算方便**。
$$
E(i) = \frac{1}{2}\sum_{k = 1}(y_k^{(i)}-o_k^{(i)})^2
$$
训练数据的（总体）平均代价：
$$
E_{total} = \frac{1}{n}\sum_{i=1}^NE(i)
$$

### 3.2 隐含层-->输出层的权值更新 

主要通过==链式法则求解==。

比如说我们要$w_5$对整体误差产生的影响，则：
$$
\frac{\partial E_{total}}{\partial w_5} = \frac{\partial E_{total}}{\partial out_{o1}}*\frac{\partial out_{o1}}{\partial net_{o1}}*\frac{\partial net_{o1}}{\partial w_5}
$$

1. 这个时候需要一个一个求了，首先第一个$\frac{\partial E_{total}}{\partial out_{o1}}$:

$$
\frac{\partial E_{total}}{\partial out_{o1}} = out_1-target_1
$$

2. 然后计算第二个$\frac{\partial out_{o1}}{\partial net_{o1}}$:

$$
\frac{\partial w_{out_1}}{\partial net_{o1}} = out_1(1-out_1)
$$

*注意：其实这里就相当于sigmoid的导数*

3. 然后计算第三个$\frac{\partial net_{o1}}{\partial w_5}$:

$$
\frac{\partial net_{o1}}{\partial w_5}= out_{h1}
$$

*注意：这里的$out_{h1}$是前面前向传播得到的*

4. 最后三者相乘：
   $$
   \frac{\partial E_{total}}{\partial w_5} =(out_1-target_1)*out_1(1-out_1)*out_{h1}
   $$

5. 最后基于梯度下降

$$
w_5 = w_5-\alpha*\frac{\partial E_{total}}{\partial w_5}
$$

其中，$\alpha$为学习率。

在这里，我们将
$$
\delta_{0}=\frac{\partial E_{total}}{\partial w_{out_1}}*\frac{\partial out_{o1}}{\partial net_{o1}}
$$
那么
$$
\frac{\partial E_{total}}{\partial w_5}=\delta_{0}*out_{h_1}
$$

### 3.3 隐含层-->隐含层的权值更新

<img src="C:\Users\49252\AppData\Roaming\Typora\typora-user-images\image-20201216104430953.png" alt="image-20201216104430953" style="zoom:67%;" />
$$
\frac{\partial E_{total}}{\partial w_1} = \frac{\partial E_{total}}{\partial out_{h1}}*\frac{\partial out_{h1}}{\partial net_{h1}}*\frac{\partial net_{h1}}{\partial w_1}
$$
其中，$E_{total}$是由$E_{o1}$和$E_{O2}$组成的。

因此
$$
\frac{\partial E_{total}}{\partial out_{h1}} = \frac{\partial E_{o1}}{\partial out_{h1}} +\frac{\partial E_{o2}}{\partial out_{h1}}
$$
我们可以求
$$
\frac{\partial E_{o1}}{\partial out_{h1}}  = \frac{\partial E_{o1}}{\partial out_{o1}}*\frac{\partial out_{o1}}{\partial net_{o1}}*\frac{\partial net_{o1}}{\partial out_{h1}}
$$


同理：
$$
\frac{\partial E_{o2}}{\partial out_{h1}}  = \frac{\partial E_{o2}}{\partial out_{o1}}*\frac{\partial out_{o1}}{\partial net_{o1}}*\frac{\partial net_{o1}}{\partial out_{h1}}
$$
由公式8可得：
$$
\frac{\partial E_{total}}{\partial out_{h1}} = \sum_o\delta_o*w_{ho}
$$

$$
\frac{\partial net_{o1}}{\partial out_{h1}} = w_{ho}
$$

可得：
$$
\frac{\partial E_{total}}{\partial w_1}  = \sum_o\delta_o*w_{ho}*out_{h_1}(1-out_{h_1})*i_1
$$
最后可通过梯度下降更新权重。

## 4 代码

代码如下：

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidDerivationx(y):
    return y * (1 - y)


if __name__ == '__main__':
    # 初始化一些参数
    alpha = 0.5
    numIter = 1000000 #迭代次数
    w1 = [[0.15, 0.20], [0.25, 0.30]]  # Weight of input layer
    w2 = [[0.40, 0.45], [0.50, 0.55]]
    # print(np.array(w2).T)
    b1 = 0.35
    b2 = 0.60
    x = [0.05, 0.10]
    y = [0.01, 0.99]
    # 前向传播
    z1 = np.dot(w1, x) + b1     # dot函数是常规的矩阵相乘
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    for n in range(numIter):
        # 反向传播 使用代价函数为C=1 / (2n) * sum[(y-a2)^2]
        # 分为两次
        # 一次是最后一层对前面一层的错误

        delta2 = np.multiply(-(y-a2), np.multiply(a2, 1-a2))
        # for i in range(len(w2)):
        #     print(w2[i] - alpha * delta2[i] * a1)
        #计算非最后一层的错误
        # print(delta2)
        delta1 = np.multiply(np.dot(np.array(w2).T, delta2), np.multiply(a1, 1-a1))
        # print(delta1)
        # for i in range(len(w1)):
            # print(w1[i] - alpha * delta1[i] * np.array(x))
        #更新权重
        for i in range(len(w2)):
            w2[i] = w2[i] - alpha * delta2[i] * a1
        for i in range(len(w1)):
            w1[i] = w1[i] - alpha * delta1[i] * np.array(x)
        #继续前向传播，算出误差值
        z1 = np.dot(w1, x) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        print(str(n) + " result:" + str(a2[0]) + ", result:" +str(a2[1]))
        # print(str(n) + "  error1:" + str(y[0] - a2[0]) + ", error2:" +str(y[1] - a2[1]))
```

