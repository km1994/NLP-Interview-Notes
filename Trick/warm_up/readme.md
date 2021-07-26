# 【关于 Warm up 】那些你不知道的事

> 作者：杨夕
> 
> 论文学习项目地址：https://github.com/km1994/nlp_paper_study
> 
> 《NLP 百面百搭》地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。
> 

![](img/微信截图_20210301212242.png)

> NLP && 推荐学习群【人数满了，加微信 blqkm601 】

![](img/20210523220743.png)

- [【关于 Warm up 】那些你不知道的事](#关于-warm-up-那些你不知道的事)
  - [一、 什么是 Warm up？](#一-什么是-warm-up)
  - [二、为什么需要 Warm up？](#二为什么需要-warm-up)
  - [参考](#参考)

## 一、 什么是 Warm up？

Warmup 是在 ResNet 论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，训练了一些 epoches 或者 steps (比如 4 个 epoches,10000steps),再修改为预先设置的学习来进行训练。

## 二、为什么需要 Warm up？

- **在训练的开始阶段，模型权重迅速改变**。 刚开始模型对数据的“分布”理解为零，或者是说“均匀分布”（当然这取决于你的初始化）；在第一轮训练的时候，每个数据点对模型来说都是新的，模型会很快地进行数据分布修正，**如果这时候学习率就很大，极有可能导致开始的时候就对该数据“过拟合”，后面要通过多轮训练才能拉回来，浪费时间。**当训练了一段时间（比如两轮、三轮）后，模型已经对每个数据点看过几遍了，或者说对当前的batch而言有了一些正确的先验，较大的学习率就不那么容易会使模型学偏，所以可以适当调大学习率。这个过程就可以看做是warmup。那么为什么之后还要decay呢？当模型训到一定阶段后（比如十个epoch），模型的分布就已经比较固定了，或者说能学到的新东西就比较少了。如果还沿用较大的学习率，就会破坏这种稳定性，用我们通常的话说，就是已经接近loss的local optimal了，为了靠近这个point，我们就要慢慢来。

- **mini-batch size较小，样本方差较大**。第二种情况其实和第一种情况是紧密联系的。在训练的过程中，**如果有mini-batch内的数据分布方差特别大，这就会导致模型学习剧烈波动，使其学得的权重很不稳定**，这在训练初期最为明显，最后期较为缓解（所以我们要对数据进行scale也是这个道理）。




## 参考

1. [神经网络中 warmup 策略为什么有效；有什么理论解释么？](https://www.zhihu.com/question/338066667)
2. [AdaBelief-更稳定的优化器](https://xv44586.github.io/2020/10/25/adabelief/)
3. [深度神经网络模型训练中的最新 tricks 总结【原理与代码汇总】](https://bbs.cvmart.net/articles/3320/vote_count?)
4. [【基础知识】Warmup预热学习率_菜鸟起飞-程序员宅基地](http://www.cxyzjd.com/article/nefetaria/110212564)
5. [”预热学习率“ 的搜索结果](http://www.cxyzjd.com/searchArticle?qc=%E9%A2%84%E7%83%AD%E5%AD%A6%E4%B9%A0%E7%8E%87&page=1)
6. [ICML 2020 | 摆脱warm-up！巧置LayerNorm使Transformer加速收敛](https://www.msra.cn/zh-cn/news/features/pre-ln-transformer)
7. [深度學習Warm up策略在幹什麼?](https://chih-sheng-huang821.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92warm-up%E7%AD%96%E7%95%A5%E5%9C%A8%E5%B9%B9%E4%BB%80%E9%BA%BC-95d2b56a557f)
8. [深度学习深度学习模型训练的tricks总结](https://www.codenong.com/cs105809498/)




