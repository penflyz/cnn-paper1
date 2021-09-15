# 残差网络（ResNet）

## 参考资料

知乎：[神经网络训练中的梯度消失与梯度爆炸](https://zhuanlan.zhihu.com/p/25631496)

知乎：[残差网络解决了什么，为什么有效？](https://zhuanlan.zhihu.com/p/80226180)

# 1 动机：深度学习的“两朵乌云”

**致敬物理界的两朵乌云： 第一朵乌云出现在光的波动理论上。 第二朵乌云出现在关于能量均分的麦克斯韦-玻尔兹曼理论上。 **

### 1.1 第一朵乌云：梯度消失/爆炸

* 梯度消失

通俗来说，当神经网络的层数过深且梯度过小时，**接近输出层的隐含层相对正常，通过反向传播会导致接近输入层的梯度会消失，进而导致输入层权重更新缓慢或停滞。**

<img src="https://pic2.zhimg.com/80/v2-da5606a2eebd4d9b6ac4095b398dacf5_720w.png" alt="img" style="zoom: 67%;" />

可见，![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%27%5Cleft%28x%5Cright%29)的最大值为![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B4%7D)，而我们初始化的网络权值![[公式]](https://www.zhihu.com/equation?tex=%7Cw%7C)通常都小于1，因此![[公式]](https://www.zhihu.com/equation?tex=%7C%5Csigma%27%5Cleft%28z%5Cright%29w%7C%5Cleq%5Cfrac%7B1%7D%7B4%7D)，因此对于上面的链式求导，层数越多，求导结果![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C%7D%7B%5Cpartial+b_1%7D)越小，因而导致梯度消失的情况出现。

对于梯度消失问题，可以考虑使用Relu代替sigmoid激活函数。

* 梯度爆炸

当梯度大于1时，由于深度网络的原因，就会反向传播导致指数爆炸。



**这个问题很大程度上已经被标准初始化和中间层正规化方法有效控制了**，这些方法使得深度神经网络可以收敛。

还需要注意的是，无论是Sigmoid还是Relu激活函数，仍旧有可能发生梯度消失或爆炸。

### 1.2 第二朵乌云：网络退化

首先要明确一点，并不是网络越深，效果越好。

当网络越来越深，神经网络的表现是先增加后饱和，然后会降低。

# 2 残差网络

恒等函数：f（x） = x

<img src="https://segmentfault.com/img/bV8FN4?w=502&h=317" alt="clipboard.png" style="zoom:67%;" />

* 从前向传播看，跳跃连接，也就是$a^{[l+2]}=g(a^{[l]}+z^{[l+2]})$。

  <img src="https://pic3.zhimg.com/80/v2-08a072672eb123f69fdf5daa6d9fc98e_720w.jpg" alt="img" style="zoom:50%;" />

* 可以从任意低层直接传播到任意高层，包含了一个天然的恒等影射，即h(x) =F(x)+x

* 反向传播时。**错误信号可以不经过任何中间权重矩阵变换直接传播到低层，一定程度上可以缓解梯度弥散问题**

  正常情况下：

  <img src="https://pic2.zhimg.com/v2-ec5ee0d5bbc9e22737b5fb917324980d_b.webp" alt="img" style="zoom: 67%;" />

  由上图可以看出反向传播得到的梯度是[0.01 0.0001]

  增加残差块之后：

  <img src="https://pic3.zhimg.com/80/v2-08a072672eb123f69fdf5daa6d9fc98e_720w.jpg" alt="img" style="zoom: 67%;" />

  当进行后向传播时，右边来自深层网络传回来的梯度为1，经过一个加法门，橙色方向的梯度为dh(x)/dx=1，蓝色方向的梯度也为1。这样，经过梯度传播后，现在传到前一层的梯度就变成了[1, 0.0001, 0.01]，多了一个“1”！**正是由于多了这条捷径，来自深层的梯度能直接畅通无阻地通过，去到上一层，使得浅层的网络层参数等到有效的训练！**

# 3 代码

resnet1

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100):  # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # output: [b, 512, h, w],
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [b, c]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
```

resnet_1:

```
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
from resnet1 import resnet18
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    # [-1~1]
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
print(np.shape(y))
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(512)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(500):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                logits = model(x)
                # [b] => [b, 100]
                y_onehot = tf.one_hot(y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
```

