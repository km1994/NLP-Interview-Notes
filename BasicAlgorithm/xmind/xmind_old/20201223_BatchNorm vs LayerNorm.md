### BatchNorm vs LayerNorm

**BatchNorm是什么？为什么有用？存在什么问题？**

**LayerNorm是什么？为什么有用？存在什么问题？**

**BatchNorm和LayerNorm的区别是什么？**

![Normalization](img/Normalization.png)

【动机】

ICS问题：由于深度神经网络的叠加，容易导致每一层的输入数据发生变化，这种问题会随着层数的增加而加剧，使得高层需要不断去重新适应底层的参数更新。

带来的后果：

1. 上层参数需要不断适应新的输入数据分布，导致学习速度下降
2. 下层输入的变化可能趋于变大或变小，导致上层落入饱和区，从而学习过早停止
3. 每层的更新都会影响到其他层，因此每层参数更新策略需要尽可能谨慎

【BatchNorm】就是在神经元归一化之后增加两个调节参数。具体操作如下图所示：

![BatchNorm](img/BatchNorm操作.png)

batchnorm降低了数据之间的绝对差异，有一个去相关的性质，更多的考虑相对差异性，因此在分类任务上具有更好的效果。

问题: 

1. BN特别依赖Batch Size；当Batch size很小的适合，BN的效果就非常不理想了。在很多情况下，Batch size大不了，因为你GPU的显存不够。所以，通常会有其他比较麻烦的手段去解决这个问题，比如MegDet的CGBN等；
2. BN对处理序列化数据的网络比如RNN是不太适用的；So，BN的应用领域减少了一半。
3. BN只在训练的时候用，inference的时候不会用到，因为inference的输入不是批量输入。

【LayerNorm】是对输入批次的每一个样本独立进行规范化，对于文字序列，因为有embedding层，所以输入的tensor通常为(bsz,seqlen,emb_dim)，其规范化一般只在最后一维度进行（transformer和bert）。

LN相比于BN不受batch_size的影响，同时LN可以很好地用到序列型网络，同时LR在训练过程和inference的过程中都可以存在。

问题: LN的性能干不过BN

【区别】主要是normalization方向不同

- BN：沿着batch中特征方向进行normalization

- LN：对一个样本的所有特征进行normalization。当特征量纲不同时会有问题。

   ![BN方向](img/BN在神经网络方向示意图.png)

   ![LN方向](img/LN在神经网络方向示意图.png)



