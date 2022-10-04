# 【关于BERT在输入层引入额外特征】 那些你不知道的事

> 作者：杨夕
> 
> 介绍：研读顶会论文，复现论文相关代码
> 
> NLP 百面百搭 地址：https://github.com/km1994/NLP-Interview-Notes
> 
> 推荐系统 百面百搭 地址：https://github.com/km1994/RES-Interview-Notes
>
> 搜索引擎 百面百搭 地址：https://github.com/km1994/search-engine-Interview-Notes 【编写ing】
> 
> NLP论文学习笔记：https://github.com/km1994/nlp_paper_study
> 
> 推荐系统论文学习笔记：https://github.com/km1994/RS_paper_study
> 
> GCN 论文学习笔记：https://github.com/km1994/GCN_study
> 
> **推广搜 军火库**：https://github.com/km1994/recommendation_advertisement_search
![](other_study/resource/pic/微信截图_20210301212242.png)

> 手机版笔记，可以关注公众号 **【关于NLP那些你不知道的事】** 获取，并加入 【NLP && 推荐学习群】一起学习！！！

> 注：github 网页版 看起来不舒服，可以看 **[手机版NLP论文学习笔记](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=1&sn=14d34d70a7e7cbf9700f804cca5be2d0&chksm=1bbff26d2cc87b7b9d2ed12c8d280cd737e270cd82c8850f7ca2ee44ec8883873ff5e9904e7e&scene=18#wechat_redirect)**

## 一、动机



## 二、方法


1. 邱锡鹏老师的flat : 他简单来说呢，是把词典拼在原文的输入的后面，然后和前面的词典的位置共享一个position的embedding。相当于呢是词典对原文进行了一种提示，引入了外部的知识。
2. 在底层增加一个embedding层，就像bert原生的一样，从input-id-embedding、token-type-embedding和mask-embedding变成input-id-embedding、token-type-embedding、mask-embedding和keyword-embedding。输入，lookup之后，输出进行加和即可。这样的话，只需要随机初始化keyword-embedding就行，其他参数都可以加载原始参数;
3. 在关键词前后加上特殊的标识符，让模型强行去学习，关键词的信息。


## 参考

1. [BERT在输入层如何引入额外特征？](https://www.zhihu.com/question/470540809/answer/2691234148)