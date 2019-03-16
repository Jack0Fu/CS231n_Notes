#### Lecture 6 Training Neural Networks, Part I

![1552528184148](assets/1552528184148.png)

![1552528206728](assets/1552528206728.png)

#####  1. Activation Functions

![1552567890229](assets/1552567890229.png)

###### 1.1 Sigmoid

sigmoid现在基本上都不使用了，可能它在早期历史的舞台上出现过，我们总是喜欢找个不太好的东西来体现出好东西的牛逼之处。Sigmoid函数的缺点很明显：1. S型曲线两端趋向于直线，梯度几乎为0，导致反向传播时梯度消失的问题；2. Sigmoid函数，无论输入什么数值，输出值都为正，然后导致W的梯度值均为正或者负，就是说W只能朝一个方向调整了，导致拟合性变差（==TODO：为啥呢==）；3. 指数计算开销太大

![1552568057841](assets/1552568057841.png)

![1552569119735](assets/1552569119735.png)

==TODO==:

为啥激活函数sigmoid的non-zero centered不行呢？这里面的机制我还是不太懂

###### 1.2 tanh(x)

相比于Sigmoid，tanh(x)有了一点提高，输出值关于0对称。但是，曲线两端仍趋向于直线，同样会导致梯度消失。

![1552569136529](assets/1552569136529.png)

###### 1.3 ReLU （Rectified Linear Unit）

ReLU: $f(x) = max(0,x)​$

ReLU 是根据神经科学中对人体神经的观测得出的一个近似函数，优点：1. 提高了Loss的收敛速度；2. 引入稀疏性，人体神经系统处理某个特定信息，只需要激活部分神经元即可，不需要将所有的神经元都激活，ReLU正好也能做到这一点

reference: ReLU导致神经元死亡的解释：https://www.zhihu.com/question/67151971

原作论文：Deep Sparse Rectifier Neural Networks http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf

![1552569333025](assets/1552569333025.png)

![1552570816078](assets/1552570816078.png)

###### 1.4 Leaky ReLU

相比于ReLU，Leaky ReLU在负区间做了改进，梯度在负区间也不会消失

![1552572049869](assets/1552572049869.png)

###### 1.5 ELU 

有指数运算，开销大

![1552572097292](assets/1552572097292.png)

######  1.6 Maxout "Neuron"

这种方法综合ReLU和Leaky ReLU，正负区间都是线性的，不会出现饱和的情况，梯度不会消失。但是带来的缺点是，参数量增大。

![1552572139228](assets/1552572139228.png)



##### 2. Data Preprocessing

预处理数据，将原始数据变成均值为0，方差为1的标准正态分布。

![1552612805588](assets/1552612805588.png)



在图像预处理中，zero-centered data所取得平均值，有两种取法：1、取一整张图片（32x32x3）的平均像素值作为平均值；2、取每个通道（rgb）的像素平均值。

![1552612899977](assets/1552612899977.png)

在实际应用中，我们也会对原始数据进行主成分分析（PCA）和白化处理（Whiten）。这种情况下一般不会对数据进行方差标准化，因为方差标准化会破坏掉一些特征。

![1552613117259](assets/1552613117259.png)

#####  3. Weight Initialization

权重参数如何初始化？这是一个必须要考虑的问题。

我们第一个想到是随机初始化，让权重以正态分布的概率随机选一个值，这种方法对小网络还okay，但是对更深的网络就不太行了。

（人类总希望能够控制一切，对权重也不例外，我们不能任由它随意发展，而是要控制权重朝着好的方向发展）

![1552614380138](assets/1552614380138.png)

其他的思路就有很多了，这里列举了一种： Reasonable initialization

![1552615059178](assets/1552615059178.png)

这里列举了多篇研究weight initialization的论文：

![1552614743531](assets/1552614743531.png)

==初始化方法？==

##### 4. Batch Normalization

reference:Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift   https://arxiv.org/abs/1502.03167

![1552615205904](assets/1552615205904.png)

![1552615232640](assets/1552615232640.png)

![1552615263065](assets/1552615263065.png)

![1552615282287](assets/1552615282287.png)

![1552615329848](assets/1552615329848.png)

![1552615343961](assets/1552615343961.png)

reference : https://www.cnblogs.com/guoyaohua/p/8724433.html

深度神经网络, 随着网络深度加深，训练起来越困难，收敛越来越慢。** BN要解决的正是这个问题。

简单的说，BN是用在哪里的？BN是用在全连接层之后，处理好全连接层输出的数据后，再输入到非线性激活函数中。

BN有啥好处呢？用了BN，Loss的收敛速度就加快了很多，训练速度加快了。而且，减少了对参数初始化的依赖，不用太管那些参数初始化，可以调高learning_rate。

BN具体怎么用呢？在训练过程中，来一个mini-batch， 先求出mini-batch mean 和mini-batch variance, 然后做一下normalize，就得到一个标准正态分布。然后再做一下scale and shift，这里增加了两个参数，而且这两个参数是通过学习不断变化来得到的，提高了网络的表达能力。

==在test 中，mean和std是使用训练时的数据？？==

##### 5. Babysitting the learning process

这里讲了在这个训练过程中，我们怎么一步一步实现自己的目标。

第一步，预处理数据；

第二步，选择一个网络结构；

第三步，先不用正则项（disable regularization），看一下loss是多少，合不合理

第四步，试一下加大正则项，看一下loss有没有降下来

第五步，选择一个小正则项，试着调learning_rate ，让loss下降

![1552618807764](assets/1552618807764.png)

![1552618834986](assets/1552618834986.png)

![1552619421573](assets/1552619421573.png)

##### 6. Hyperparameter Optimization

超参数优化策略：交叉验证

![1552619778827](assets/1552619778827.png)

先做粗略的搜索，再做更精细的搜索，最好是在log space 中搜索。

两种搜索方式的对比，当然选择random search啊。

![1552620170588](assets/1552620170588.png)

需要调节的超参数：

![1552619624990](assets/1552619624990.png)

不同学习速率对比图：

![1552619637373](assets/1552619637373.png)

首先怀疑是不是初始化错误：

![1552620221707](assets/1552620221707.png)

![1552620253243](assets/1552620253243.png)

![1552620270351](assets/1552620270351.png)

调参真的是深度学习训练过程中的一个体力活，但是调的多了自然就总结出来一些经验，能够更快地找到比较好的参数。

##### 7. Summary 

![1552619681924](assets/1552619681924.png)

![1552619732604](assets/1552619732604.png)