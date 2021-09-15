# Inception network

参考文章：[inception网络模型](https://www.cnblogs.com/dengshunge/p/10808191.html)

​					[吴恩达深度学习课程](https://www.bilibili.com/video/BV1F4411y7o7?p=18)

# 1 inception网络的提出

提升网络性能，一般最直接的方法是提高网络深度和宽度，但是缺点就是会增大训练参数和过拟合。

解决的办法==引入系数特性和全连接层转换成稀疏连接。==但是现在的计算框架对非均匀的稀疏数据进行计算是非常低效的，主要是因为查找和缓存的开销。（*这部分需要仔细看一下*）

有大量文献指出，将稀疏矩阵聚类成相对密集的子矩阵，能提高计算性能。根据此想法，提出了Inception结构。

# 2 inception网络结构

## 2.1 1*1卷积

2013年提出。

适用于==多channel信道==的图像，不是6x6x1而是6x6x32

<img src="C:\Users\49252\AppData\Roaming\Typora\typora-user-images\image-20201222140133897.png" alt="image-20201222140133897" style="zoom:50%;" />

**其本质是1x1为了不改变图片的长和宽，然后用卷积的filter压缩调整channel**

如上图，当我们需要将28X28X192的图像变为28X28X32，我们只需要32个1X1X192的滤波器。

同时这里就像把一个切片做非线性处理映射到一个点上，所以1X1卷积又叫做network in network。

## 2.2 inception结构

<img src="Inception network.assets/1456303-20190504162327115-2119269106.png" alt="img" style="zoom:50%;" />

* 采用不同的卷积核意味着不同的感受野，最后的拼接意味着不同尺度特征的融合
* 在inception结构中，大量采用了1x1的矩阵，主要是两点作用：1）对数据进行降维；2）引入更多的非线性，提高泛化能力，因为卷积后要经过ReLU激活函数。

## 2.3 GoogLeNet网络

GoogLeNet是由inception模块进行组成的，结构太大了，就不放出来了，这里做出几点说明：

　　a）GoogLeNet采用了模块化的结构，方便增添和修改；

　　b）网络最后采用了average pooling来代替全连接层，想法来自NIN,事实证明可以将TOP1 accuracy提高0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便以后大家finetune;

　　c）虽然移除了全连接，但是网络中依然使用了Dropout；

　　d）为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度。文章中说这两个辅助的分类器的loss应该加一个衰减系数，但看源码中的model也没有加任何衰减。此外，实际测试的时候，这两个额外的softmax会被去掉。

## 2.4 其他

还有后续的inception结构，这里不详细写了。

