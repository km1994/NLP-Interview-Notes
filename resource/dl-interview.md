# 深度学习面试

## 目录

- [深度学习面试](#深度学习面试)
	- [目录](#目录)
	- [为什么要激活函数？而且还是非线性的激活函数？没了激活函数会怎样？](#为什么要激活函数而且还是非线性的激活函数没了激活函数会怎样)
	- [优化算法](#优化算法)
	- [BP算法](#bp算法)
	- [CNN](#cnn)
	- [RNN](#rnn)
	- [LSTM](#lstm)
	- [GRU](#gru)
	- [Batch Normalization：](#batch-normalization)
	- [Dropout](#dropout)
	- [语言模型](#语言模型)
	- [word2vec](#word2vec)
	- [Transformer](#transformer)
	- [Bert](#bert)
	- [XLNet](#xlnet)
	- [词向量演进历史](#词向量演进历史)

##  为什么要激活函数？而且还是非线性的激活函数？没了激活函数会怎样？
如果使用线性函数，每一层输出都是上层输入的线性函数，无论神经网络有多少层，输出都是输入的线性组合，加深神经网络的层数就没有什么意义了。使用线性激活函数时，无法处理非线性问题，相反如果使用非线性的激活函数，根据通用近似定理 (Universal approximation theorem），只需一个包含足够多神经元的隐层，多层前馈神经网络就能以任意精度逼近任意复杂度的连续函数。
- Sigmoid函数：
$$\sigma(x)=\frac{1}{1+\mathrm{e}^{-x}}$$	
	- 优点：平滑易求导；输出值可以作为概率，具备可解释性
	- 缺点：导函数$0\leq\sigma(x)^{\prime}=\sigma(x)(1-\sigma(x))\leq \frac{1}{4}$，反向传播易导致梯度消失；含有指数运算较为耗时；输出值不以0为中心，可能导致模型收敛速度慢；
- Tanh函数：
$$\tanh(x)=2 \sigma(2 x)-1=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$
	- 优点：平滑易求导；输出值以0为中心，模型收敛快
	- 缺点：导函数$0\leq\tanh(x)^{\prime}=1-\tanh(x)^2\leq 1$，反向传播易导致梯度消失；含有指数运算较为耗时；
- ReLU函数：
$$\max(0,x)$$
	- 优点：在正区间上解决了梯度消失的问题；计算简单，因为只需判断是否大于0；收敛速度比上面两个激活函数要快
	- 缺点：输出值为0的概率太大，容易导致神经元死亡，无法重新激活
	- 改进方法：初始化权重时别把神经元弄死；学习率不要设置太高以防权重更新幅度太大把神经元弄死；采用动态调整学习率的优化算法；采用Leaky ReLU/PReLU函数替代
- Leaky ReLU/PReLU函数：
$$\max(\alpha x,x)$$
	- 优点：修正了ReLU函数的缺点，如果$\alpha$是固定值则为Leaky ReLU，如果$\alpha$需要自己学出来则为PReLU
- Sigmoid非零中心导致其下层神经元的所有权重更新方向一致：
$$h^t=w_1y_1^{t-1}+w_2y_2^{t-1}$$
$$y^t=f(h^t)$$
$$\cfrac{\partial Loss}{\partial w_1}=\cfrac{\partial Loss}{\partial y^t}\cdot\cfrac{\partial y^t}{\partial h^t}\cdot\cfrac{\partial h^t}{\partial w_1}=\cfrac{\partial Loss}{\partial y^t}\cdot\cfrac{\partial y^t}{\partial h^t}\cdot y_1^{t-1}$$
$$\cfrac{\partial Loss}{\partial w_2}=\cfrac{\partial Loss}{\partial y^t}\cdot\cfrac{\partial y^t}{\partial h^t}\cdot\cfrac{\partial h^t}{\partial w_2}=\cfrac{\partial Loss}{\partial y^t}\cdot\cfrac{\partial y^t}{\partial h^t}\cdot y_2^{t-1}$$
其中，$y_1^{t-1}$和$y_2^{t-1}$都是上层神经元的输出，如果上层神经元采用的是Sigmoid函数激活的话，那么$y_1^{t-1}$和$y_2^{t-1}$均恒大于0，此时$w_1,w_2$的梯度正负号恒一致，从而也就导致权重更新方向一致。不过这是只计算单个样本损失就更新梯度的情形，通常训练都是采用mini-batch梯度下降，所以会同时算多个样本的损失，然后累加所有样本损失给出的梯度更新值以后作为最终梯度更新值，此时更新时不一定方向是一致的，比如：
$$y_1^t:\Delta w_1=+6,\Delta w_2=+0.5$$
$$y_2^t:\Delta w_1=-0.2,\Delta w_2=-3$$
那么$w_1$和$w_2$总的梯度更新幅度为
$$\Delta w_1=+5.8,\Delta w_2=-2.5$$
显然两者的更新方向此时不再一致
- ReLU神经元死亡导致权重无法更新分析：
$$h^t=w_1y_1^{t-1}+w_2y_2^{t-1}$$
$$y^t=\text{ReLU}(h^t)$$
如果此时第$t$层神经元死亡的话，也即$y^t$恒等于0，那么
$$\cfrac{\partial Loss}{\partial w_1}=\cfrac{\partial Loss}{\partial y^t}\cdot\cfrac{\partial y^t}{\partial h^t}\cdot\cfrac{\partial h^t}{\partial w_1}=\cfrac{\partial Loss}{\partial y^t}\cdot 0 \cdot y_1^{t-1}=0$$
$$\cfrac{\partial Loss}{\partial w_2}=\cfrac{\partial Loss}{\partial y^t}\cdot\cfrac{\partial y^t}{\partial h^t}\cdot\cfrac{\partial h^t}{\partial w_2}=\cfrac{\partial Loss}{\partial y^t}\cdot 0 \cdot y_2^{t-1}=0$$
显然，此时第$t$层神经元的权重永远无法更新，从而也就导致永远无法激活
- 梯度消失的解决办法：BN；更换激活函数
- 梯度爆炸的解决办法：BN；梯度剪切；
- 交叉熵损失函数本身是一个凸函数，但是为什么神经网络的损失函数非凸？
交叉熵损失函数为凸仅仅是对于Logistic回归和Softmax而言，对于神经网络来说显然不是，毕竟神经网络待学习的参数不止最后一层分类层，其它层也需要学习，而且对称翻转神经网络模型不变，损失函数值不变，但是参数的位置改变了，所以即使有最优解也不止一个，具体参见：https://www.zhihu.com/question/265516791/answer/514198101
## 优化算法
- 整体叙述框架参见：https://zhuanlan.zhihu.com/p/32230623 ，具体内容参见：https://blog.csdn.net/u010089444/article/details/76725843 和 https://ruder.io/optimizing-gradient-descent/index.html
- 基本框架：定义当前时刻待优化参数为$\theta_t\in R^{d}$，损失函数为$J(\theta)$，学习率为$\eta$，参数更新框架为：
1. 计算损失函数关于当前参数的梯度：$g_t=\nabla J(\theta_t)$；
2. 根据历史梯度计算一阶动量（一次项）和二阶动量（二次项）：$m_t=\phi(g_1,g_2,...,g_t),V_t=\psi(g_1,g_2,...,g_t)$；
3. 计算当前时刻的下降梯度：$\Delta\theta_t=-\eta\cdot\cfrac{m_t}{\sqrt{V_t}}$
4. 根据下降梯度更新参数：$\theta_{t+1}=\theta_t+\Delta\theta_t$

- SGD：由于SGD没有动量的概念，也即没有考虑历史梯度，所以当前时刻的动量即为当前时刻的梯度$m_t=g_t$，且二阶动量$V_t=E$，所以SGD的参数更新公式为
$$\Delta\theta_t=-\eta\cdot g_t$$
$$\theta_{t+1}=\theta_t-\eta\cdot g_t$$
缺点：下降速度慢，而且可能会在沟壑（还有鞍点）的两边持续震荡，停留在一个局部最优点。
- SGD with Momentum：为了抑制SGD的震荡，SGDM认为梯度下降过程可以加入惯性。下坡的时候，如果发现是陡坡，那就利用惯性跑的快一些。SGDM全称是SGD with momentum，在SGD基础上引入了一阶动量。而所谓的一阶动量就是该时刻梯度的指数加权移动平均值：$\eta\cdot m_t:=\beta\cdot m_{t-1}+\eta\cdot g_t$（其中当前时刻的梯度$g_t$并不严格按照指数加权移动平均值的定义采用权重$1-\beta$，而是使用我们自定义的学习率$\eta$），那么为什么要用移动平均而不用历史所有梯度的平均？因为移动平均存储量小，且能近似表示历史所有梯度的平均。由于此时仍然没有二阶动量，所以$V_t=E$，那么SGDM的参数更新公式为
$$\Delta\theta_t=-\eta\cdot m_t=-\left(\beta m_{t-1}+\eta g_t\right)$$
$$\theta_{t+1}=\theta_t-\left(\beta m_{t-1}+\eta g_t\right)$$
所以，当前时刻参数更新的方向不光取决于当前时刻的梯度，还取决于之前时刻的梯度，特别地，当$\beta=0.9$时，$m_t$近似表示的是前10个时刻梯度的指数加权移动平均值，而且离得越近的时刻的梯度权重也越大。
优点：利用历史梯度作为惯性克服了SGD可能会在沟壑的两边持续震荡，停留在一个局部最优点的缺点，同时还加速了收敛。
缺点：对于比较深的沟壑有时用Momentum也没法跳出
	- 指数加权移动平均值（exponentially weighted moving average，EWMA）：假设$v_{t-1}$是$t-1$时刻的指数加权移动平均值，$\theta_t$是$t$时刻的观测值，那么$t$时刻的指数加权移动平均值为
$$\begin{aligned}
v_t&=\beta v_{t-1}+(1-\beta)\theta_t \\
&=(1-\beta)\theta_t+\sum_{i=1}^{t-1}(1-\beta)\beta^i\theta_{t-i}
\end{aligned}$$
其中$0 \leq \beta < 1,v_0=0$。显然，由上式可知，$t$时刻的指数加权移动平均值其实可以看做前$t$时刻所有观测值的加权平均值，除了第$t$时刻的观测值权重为$1-\beta$外，其他时刻的观测值权重为$(1-\beta)\beta^i$。由于通常对于那些权重小于$\frac{1}{e}$的观测值可以忽略不计，所以忽略掉那些观测值以后，上式就可以看做在求加权移动平均值。那么哪些项的权重会小于$\frac{1}{e}$呢？由于
$$\lim_{n \rightarrow +\infty}  \left(1-\frac{1}{n}\right)^n = \frac{1}{e} \approx 0.3679$$
若令$n=\frac{1}{1-\beta}$，则
$$\lim_{n \rightarrow +\infty}  \left(1-\frac{1}{n}\right)^n =\lim_{\beta \rightarrow 1}  \left(\beta\right)^{\frac{1}{1-\beta}}=\frac{1}{e} \approx 0.3679$$
所以，当$\beta\rightarrow 1$时，那些$i\geq\frac{1}{1-\beta}$的$\theta_{t-i}$的权重$(1-\beta)\beta^i$一定小于$\frac{1}{e}$。代入计算可知，那些权重小于$\frac{1}{e}$的观测值就是近$\frac{1}{1-\beta}$个时刻之前的观测值。例如当$t=20,\beta=0.9$时，$\theta_1,\theta_2,..,\theta_9,\theta_{10}$的权重都是小于$\frac{1}{e}$的，因此可以忽略不计，那么此时就相当于在求$\theta_11,\theta_12,..,\theta_19,\theta_{20}$这最近10个时刻的加权移动平均值。所以指数移动平均值可以近似看做在求最近$\frac{1}{1-\beta}$个时刻的加权移动平均值，$\beta$常取$\geq 0.9$。由于当$t$较小时，指数加权移动平均值的偏差较大，所以通常会加上一个修正因子$1-\beta^t$，加了修正因子后的公式为
$$v_t=\cfrac{\beta v_{t-1}+(1-\beta)\theta_t}{1-\beta^t} \\$$
显然，当$t$很小时，修正因子$1-\beta^t$会起作用，当$t$足够大时$(1-\beta^t)\rightarrow 1$，修正因子会自动退场。详见：https://zhuanlan.zhihu.com/p/32335746
- SGD with Nesterov Acceleration：除了利用惯性跳出局部沟壑以外，我们还可以尝试往前看一步。想象一下你走到一个盆地，四周都是略高的小山，你觉得没有下坡的方向，那就只能待在这里了。可是如果你爬上高地，就会发现外面的世界还很广阔。因此，我们不能停留在当前位置去观察未来的方向，而要向前一步、多看一步、看远一些。NAG全称Nesterov Accelerated Gradient，是在SGD、SGD-M的基础上的进一步改进，改进点在于当前时刻梯度的计算，我们知道在时刻t的主要下降方向是由累积动量决定的，自己的梯度方向说了也不算，那与其看当前梯度方向，不如先看看如果跟着累积动量走了一步，那个时候再怎么走。也即在Momentum的基础上将当前时刻的梯度$g_t$换成下一时刻的梯度$\nabla J(\theta_t-\beta m_{t-1})$，由于此时也没有考虑二阶动量，所以$V_t=E$，NAG的参数更新公式为
$$\Delta\theta_t=-\eta\cdot m_t=-\left(\beta m_{t-1}+\eta\nabla J(\theta_t-\beta m_{t-1})\right)$$
$$\theta_{t+1}=\theta_t-\left(\beta m_{t-1}+\eta\nabla J(\theta_t-\beta m_{t-1})\right)$$
优点：在Momentum的基础上进行了改进，比Momentum更具有前瞻性，除了利用历史梯度作为惯性来跳出局部最优的沟壑以外，还提前走一步看看能否直接跨过沟壑。
- AdaGrad：此前我们都没有用到二阶动量。二阶动量的出现，才意味着“自适应学习率”优化算法时代的到来。SGD及其变种以同样的学习率更新每个维度的参数（因为$\theta_t$通常是向量），但深度神经网络往往包含大量的参数，这些参数并不是总会用得到（想想大规模的embedding）。对于经常更新的参数，我们已经积累了大量关于它的知识，不希望被单个样本影响太大，希望学习速率慢一些；对于偶尔更新的参数，我们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，即学习速率大一些。因此，AdaGrad则考虑对于不同维度的参数采用不同的学习率，具体的，对于那些更新幅度很大的参数，通常历史累计梯度的平方和会很大，相反的，对于那些更新幅度很小的参数，通常其累计历史梯度的平方和会很小（具体图示参见：https://zhuanlan.zhihu.com/p/29920135 ）。所以在一个固定学习率的基础上除以历史累计梯度的平方和就能使得那些更新幅度很大的参数的学习率变小，同样也能使得那些更新幅度很小的参数学习率变大，所以AdaGrad的参数更新公式为
$$v_{t,i}=\sum_{t=1}^{t}g_{t,i}^2$$
$$\Delta\theta_{t,i}=-\frac{\eta}{\sqrt{v_{t,i}+\epsilon}}g_{t,i}$$
$$\theta_{t+1,i}=\theta_{t,i}-\frac{\eta}{\sqrt{v_{t,i}+\epsilon}}g_{t,i}$$
其中$g_{t,i}^2$表示第$t$时刻第$i$维度参数的梯度值，$\epsilon$是防止分母等于0的平滑项（常取一个很小的值$1e-8$）。显然，此时上式中的$\frac{\eta}{\sqrt{v_{t,i}+\epsilon}}$这个整体可以看做是学习率，分母中的历史累计梯度值$v_{t,i}$越大的参数学习率越小。上式仅仅是第$t$时刻第$i$维度参数的更新公式，对于第$t$时刻的所有维度参数的整体更新公式为
$$V_{t}=\operatorname{diag}\left(v_{t,1},v_{t,2},...,v_{t,d}\right)\in R^{d\times d}$$
$$\Delta\theta_{t}=-\frac{\eta}{\sqrt{V_{t}+\epsilon}}\odot g_t$$
$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{V_{t}+\epsilon}}\odot g_t$$
注意，由于$V_t$是对角矩阵，所以上式中的$\epsilon$只用来平滑$V_t$对角线上的元素。
优点：不再对所有维度的参数采用同一个固定的学习率，而是考虑对不同更新幅度的参数采用不同的学习率；
缺点：随着时间步的拉长，历史累计梯度平方和$v_{t,i}$会越来越大，这样会使得所有维度参数的学习率都不断减小（单调递减），无论更新幅度如何。而且，计算历史累计梯度平方和时需要存储所有历史梯度，而通常神经网络的参数不仅多维度还高，因此存储量巨大。
- RMSProp/AdaDelta：由于AdaGrad单调递减的学习率变化过于激进，我们考虑一个改变二阶动量计算方法的策略：不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度，采用Momentum中的指数加权移动平均值的思路。这也就是AdaDelta名称中Delta的来历。首先看最简单直接版的RMSProp，RMSProp就是在AdaGrad的基础上将普通的历史累计梯度平方和换成历史累计梯度平方和的指数加权移动平均值，所以只需将AdaGrad中的$v_{t,i}$的公式改成指数加权移动平均值的形式即可，也即
$$v_{t,i}=\beta v_{t-1,i}+(1-\beta)g_{t,i}^2$$
$$V_{t}=\operatorname{diag}\left(v_{t,1},v_{t,2},...,v_{t,d}\right)\in R^{d\times d}$$
$$\Delta\theta_{t}=-\frac{\eta}{\sqrt{V_{t}+\epsilon}}\odot g_t$$
$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{V_{t}+\epsilon}}\odot g_t$$
而AdaDelta除了对二阶动量计算指数加权移动平均以外，还对当前时刻的下降梯度$\Delta\theta_{t}$也计算一个指数加权移动平均，具体地
$$\operatorname{E}[\Delta\theta^2]_{t,i}=\gamma\operatorname{E}[\Delta\theta^2]_{t-1,i}+(1-\gamma)\Delta\theta^2_{t,i}$$
由于$\Delta\theta^2_{t,i}$目前是未知的，所以只能用$t-1$时刻的指数加权移动平均来近似替换，也即
$$\operatorname{E}[\Delta\theta^2]_{t-1,i}=\gamma\operatorname{E}[\Delta\theta^2]_{t-2,i}+(1-\gamma)\Delta\theta^2_{t-1,i}$$
除了计算出$t-1$时刻的指数加权移动平均以外，AdaDelta还用此值替换我们预先设置的学习率$\eta$，因此，AdaDelta的参数更新公式为
$$v_{t,i}=\beta v_{t-1,i}+(1-\beta)g_{t,i}^2$$
$$V_{t}=\operatorname{diag}\left(v_{t,1},v_{t,2},...,v_{t,d}\right)\in R^{d\times d}$$
$$\operatorname{E}[\Delta\theta^2]_{t-1,i}=\gamma\operatorname{E}[\Delta\theta^2]_{t-2,i}+(1-\gamma)\Delta\theta^2_{t-1,i}$$
$$\Theta_{t}=\operatorname{diag}\left(\operatorname{E}[\Delta\theta^2]_{t-1,1},\operatorname{E}[\Delta\theta^2]_{t-1,2},...,\operatorname{E}[\Delta\theta^2]_{t-1,d}\right)\in R^{d\times d}$$
$$\Delta\theta_{t}=-\frac{\sqrt{\Theta_{t}+\epsilon}}{\sqrt{V_{t}+\epsilon}}\odot g_t$$
$$\theta_{t+1}=\theta_{t}-\frac{\sqrt{\Theta_{t}+\epsilon}}{\sqrt{V_{t}+\epsilon}}\odot g_t$$
显然，对于AdaDelta算法来说，已经不需要我们自己预设学习率$\eta$了，只需要预设$\beta$和$\gamma$这两个指数加权移动平均值的衰减率即可。
优点：和AdamGrad一样对不同维度的参数采用不同的学习率，同时还改进了AdamGrad的梯度不断累积和需要存储所有历史梯度的缺点（因为移动平均不需要存储所有历史梯度）。特别地，对于AdaDelta还废除了预设的学习率，当然效果好不好还是需要看实际场景。
- Adam：谈到这里，Adam和Nadam的出现就很自然而然了——它们是前述方法的集大成者。我们看到，SGDM在SGD基础上增加了一阶动量，AdaGrad和AdaDelta在SGD基础上增加了二阶动量。把一阶动量和二阶动量都用起来，就是Adam了——Adaptive + Momentum。具体地，首先计算一阶动量
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$
然后类似AdaDelta和RMSProp计算二阶动量
$$v_{t,i}=\beta_2 v_{t-1,i}+(1-\beta_2)g_{t,i}^2$$
$$V_{t}=\operatorname{diag}\left(v_{t,1},v_{t,2},...,v_{t,d}\right)\in R^{d\times d}$$
然后分别加上指数加权移动平均值的修正因子
$$\begin{aligned} 
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \\ 
\hat{v}_{t,i} &= \dfrac{v_{t,i}}{1 - \beta^t_2} \\
\hat{V}_{t}&=\operatorname{diag}\left(\hat{v}_{t,1},\hat{v}_{t,2},...,\hat{v}_{t,d}\right)\in R^{d\times d}
\end{aligned}$$
所以，Adam的参数更新公式为
$$\Delta\theta_{t}=-\frac{\eta}{\sqrt{\hat{V}_{t}+\epsilon}}\odot \hat{m}_t$$
$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{V}_{t}+\epsilon}}\odot \hat{m}_t$$
- Nadam：Adam只是将Momentum和Adaptive集成了，但是没有将Nesterov集成进来，而Nadam则是在Adam的基础上将Nesterov集成了进来，也即Nadam = Nesterov + Adam。具体思想如下：由于NAG 的核心在于，计算当前时刻的梯度$g_t$时使用了「未来梯度」$\nabla J(\theta_t-\beta m_{t-1})$。NAdam 提出了一种公式变形的思路（详见：https://zhuanlan.zhihu.com/p/32626442 ），大意可以这样理解：只要能在梯度计算中考虑到「未来因素」，就算是达到了 Nesterov 的效果。既然如此，我们就不一定非要在计算$g_t$时使用「未来因素」，可以考虑在其他地方使用「未来因素」。具体地，首先NAdam在Adam的基础上将$\hat{m}_t$展开
$$\begin{aligned} 
\theta_{t+1}&=\theta_{t}-\frac{\eta}{\sqrt{\hat{V}_{t}+\epsilon}}\odot \hat{m}_t \\
&= \theta_{t} - \frac{\eta}{\sqrt{\hat{V}_{t}+\epsilon}} \odot(\frac{\beta_1 m_{t-1}}{1 - \beta^t_1} + \dfrac{(1 - \beta_1) g_t}{1 - \beta^t_1}) \\
\end{aligned}$$
此时，如果我们将第$t-1$时刻的动量$m_{t-1}$用第$t$时刻的动量$m_{t}$近似代替的话，那么我们就引入了「未来因素」，所以将$m_{t-1}$替换成$m_{t}$即可得到Nadam的表达式
$$\begin{aligned} 
\theta_{t+1}&= \theta_{t} - \frac{\eta}{\sqrt{\hat{V}_{t}+\epsilon}} \odot(\frac{\beta_1 m_{t}}{1 - \beta^t_1} + \dfrac{(1 - \beta_1) g_t}{1 - \beta^t_1}) \\
&= \theta_{t} - \frac{\eta}{\sqrt{\hat{V}_{t}+\epsilon}} \odot(\beta_1\hat{m}_t+ \dfrac{(1 - \beta_1) g_t}{1 - \beta^t_1})
\end{aligned}$$
除了此思路之外，https://zhuanlan.zhihu.com/p/32626442 和 https://ruder.io/optimizing-gradient-descent/index.html 分别给出了两种不同的思路。
## BP算法
$$\left( \begin{array}{c} x_1 \\ x_2 \end{array} \right)\rightarrow\left\{ \begin{array}{rcl}
c_1=w_{11}x_1+w_{12}x_2+b_1,h_1=\sigma(c_1) \\
c_2=w_{21}x_1+w_{22}x_2+b_2,h_2=\sigma(c_2) \\
\end{array}\right\}\rightarrow c_3=w_{31}h_1+w_{32}h_2+b_3,h_3=\sigma(c_3)$$
$$\frac{\partial L}{\partial w_{31}}=\frac{\partial L}{\partial h_3}\frac{\partial h_3}{\partial c_3}\frac{\partial c_3}{\partial w_{31}}$$
$$\frac{\partial L}{\partial w_{11}}=\frac{\partial L}{\partial h_3}\frac{\partial h_3}{\partial c_3}\frac{\partial c_3}{\partial h_1}\frac{\partial h_1}{\partial c_1}\frac{\partial c_1}{\partial w_{11}}$$
## CNN
- 卷积层的作用？
提取图像的特征，并且卷积核的权重是可以学习的，由此可以猜测，在高层神经网络中，卷积操作能突破传统滤波器的限制，根据目标函数提取出想要的特征；“局部感知，参数共享”的特点大大降低了网络参数（相对于全连接），防止过拟合；之所以可以“参数共享”，是因为样本存在局部共同的特性。具体参见：https://zhuanlan.zhihu.com/p/41696749
- 池化层的作用？
	- 增大感受野：所谓感受野，即一个像素对应回原图的区域大小，假如没有pooling，一个3x3，步长为1的卷积，那么输出的一个像素的感受野就是3x3的区域，再加一个stride=1的3x3卷积，则感受野为5x5。假如我们在每一个卷积中间加上3x3的pooling呢？很明显感受野迅速增大，这就是pooling的一大用处。感受野的增加对于模型的能力的提升是必要的，正所谓“一叶障目则不见泰山也”。
	- 平移不变性：我们希望目标的些许位置的移动，能得到相同的结果。因为pooling不断地抽象了区域的特征而不关心位置，所以pooling一定程度上增加了平移不变性。具体参见：https://zhuanlan.zhihu.com/p/103350961
	- 降低优化难度和参数：我们可以用步长大于1的卷积来替代池化，但是池化每个特征通道单独做降采样，与基于卷积的降采样相比，不需要参数，更容易优化。全局池化更是可以大大降低模型的参数量和优化工作量。在一定程度上能防止过拟合的发生。具体参见：https://zhuanlan.zhihu.com/p/58381421
- Pooling池化操作的反向梯度传播？
CNN网络中另外一个不可导的环节就是Pooling池化操作，因为Pooling操作使得feature map的尺寸变化，假如做2×2的池化，假设那么第l+1层的feature map有16个梯度，那么第l层就会有64个梯度，这使得梯度无法对位的进行传播下去。其实解决这个问题的思想也很简单，就是把1个像素的梯度传递给4个像素，但是需要保证传递的loss（或者梯度）总和不变。根据这条原则，mean pooling和max pooling的反向传播也是不同的。mean pooling的前向传播就是把一个patch中的值求取平均来做pooling，那么反向传播的过程也就是把某个元素的梯度等分为n份分配给前一层，这样就保证池化前后的梯度（残差）之和保持不变，还是比较理解的；max pooling也要满足梯度之和不变的原则，max pooling的前向传播是把patch中最大的值传递给后一层，而其他像素的值直接被舍弃掉。那么反向传播也就是把梯度直接传给前一层某一个像素，而其他像素不接受梯度，也就是为0。具体图示参见：https://blog.csdn.net/qq_21190081/article/details/72871704

## RNN
$$h_{t}=\tanh \left(W\cdot\left[h_{t-1}, x_{t}\right]+b\right)$$
参数更新公式：
$$C_{1}=W_h\cdot h_{0}+W_x\cdot x_{1}+b\quad\quad\quad h_1=\tanh\left(C_1\right)$$
$$C_{2}=W_h\cdot h_{1}+W_x\cdot x_{2}+b\quad\quad\quad h_2=\tanh\left(C_2\right)$$
$$C_{3}=W_h\cdot h_{2}+W_x\cdot x_{3}+b\quad\quad\quad h_3=\tanh\left(C_3\right)$$
$$L_3=\frac{1}{2}(h_3-\hat{h}_3)^2$$
$$\begin{aligned}
\cfrac{\partial L_3}{\partial W_x}&=\cfrac{\partial L_3}{\partial h_3}\cfrac{\partial h_3}{\partial C_3}\cdot\left(x_3+\cfrac{\partial C_3}{\partial h_2}\cfrac{\partial h_2}{\partial C_2}\cdot\left(x_2+\cfrac{\partial C_2}{\partial h_1}\cfrac{\partial h_1}{\partial C_1}\cdot x_1\right)\right) \\
&=\cfrac{\partial L_3}{\partial h_3}\cfrac{\partial h_3}{\partial C_3}\cdot x_3+\cfrac{\partial L_3}{\partial h_3}\cfrac{\partial h_3}{\partial C_3}\cfrac{\partial C_3}{\partial h_2}\cfrac{\partial h_2}{\partial C_2}\cdot x_2+\cfrac{\partial L_3}{\partial h_3}\cfrac{\partial h_3}{\partial C_3}\cfrac{\partial C_3}{\partial h_2}\cfrac{\partial h_2}{\partial C_2}\cfrac{\partial C_2}{\partial h_1}\cfrac{\partial h_1}{\partial C_1}\cdot x_1
\end{aligned}$$
- 梯度消失分析：显然，上式中的最后一项的连乘项个数与序列长度成正比，而且其中的$0\leq\cfrac{\partial h_i}{\partial C_i}=\tanh^{\prime}\leq 1$（通常是这一项导致，也可以是其他小于0的项），所以随着连乘项$\cfrac{\partial h_i}{\partial C_i}$的增多极易导致梯度消失；
- 梯度爆炸分析：同理，最后一项中的$\cfrac{\partial C_i}{\partial h_{i-1}}=W_h$（通常是这一项导致，也可以是其他比较大的项），如果$W_h$稍微变大一点则极易通过连乘导致梯度爆炸；
- 当梯度消失发生时，参数$W_x$的更新梯度$\cfrac{\partial L_3}{\partial W_x}$由前几项主导（因为越靠后的项越接近0），所以这时候参数$W_x$只学习到了近距离的特征，同理，当梯度爆炸发生时，参数$W_x$的更新梯度由最后几项主导（因为最后几项此时爆炸了），所以此时参数$W_x$只学习到了远距离的特征。详见Towser的解释：https://www.zhihu.com/question/275856902/answer/460746068
- RNN为什么可以用？
	- 循环神经网络的通用近似定理 [Haykin, 2009]: 如果 一个完全连接的循环神经网络有足够数量的 sigmoid 型隐藏神经元，它可以以任意的准确率去近似任何一个非线性动力系统
	- 图灵完备 [Siegelmann et al., 1991]: 所有的图灵机都可以被一个由使用 Sigmoid 型激活函数的神经元构成的全连接循环网络来进行模拟。图灵完备意味着你的语言可以做到能够用图灵机能做到的所有事情，可以解决所有的可计算问题。
## LSTM
计算Cell值
$$\tilde{C}_{t}=\tanh \left(W_{C} \cdot\left[h_{t-1}, x_{t}\right]+b_{C}\right)$$
根据输入门和遗忘门来确定最终的Cell值
$$C_t=i_{t}\cdot\tilde{C}_{t}+f_{t}\cdot C_{t-1} $$
$$i_{t}=\sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]+b_{i}\right)$$
$$f_{t}=\sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]+b_{f}\right)$$
计算隐状态值
$$\tilde{h}_{t}=\tanh \left(C_{t}\right)$$
根据输出门来确定最终的隐状态值
$$h_{t}=o_{t}\cdot\tilde{h}_{t}$$
$$o_{t}=\sigma\left(W_{o}\left[h_{t-1}, x_{t}\right]+b_{o}\right)$$
参数更新公式：
- LSTM为什么比RNN的更擅长捕捉长距离依赖特征：
	- 从梯度的角度：根据前面对RNN梯度消失的分析可知，参数梯度更新公式里面的靠后几项极易发生梯度消失，而越靠后的项就代表着越远的特征，所以只要解决了靠后几项梯度消失的问题，那么自然也就解决了无法捕捉长距离依赖的问题。不过要注意的是，LSTM只是解决了在$C_{t-1}\rightarrow C_{t}$这条路径上的梯度消失问题，其他路径依旧存在。而且，LSTM 仍然有可能发生梯度爆炸。不过，由于 LSTM 的其他路径非常崎岖，和普通 RNN 相比多经过了很多次激活函数（导数都小于 1），因此 LSTM 发生梯度爆炸的频率要低得多。
	- 从信息流的角度：光从梯度的角度来解释还不够，因为最开始的LSTM是不含有遗忘门的（遗忘门的值恒为1），而且不含有遗忘门的LSTM梯度流更稳健，比带有遗忘门的LSTM解决梯度消失更彻底些，但是实验结果却表明带有遗忘门的LSTM捕捉长距离依赖特征的能力更强，所以除了梯度以外LSTM还有更好的解释，例如从信息流的角度。RNN的Cell也就是它的隐状态$h_t$一直在照单全收前一时刻的信息，并不是有选择性地学习，而LSTM则是有选择性地摘取当前对自己有用的信息。Towser在这个回答（https://www.zhihu.com/question/34878706/answer/665429718 ）的最后以及下面的评论区也提到了这点；
	- 从信息选择性的角度：选择性体现在，想让信息流动的时候的就让它流动，不想让它流动的时候就关掉。例如做情感分析时，只让有情感极性的词和关联词等信息输入进网络，把别的忽略掉。这样一来，网络需要记忆的内容更少，自然也更容易记住。同样以 LSTM 为例，如果某个时刻 forget gate 是 0，虽然把网络的记忆清空了、回传的梯度也截断掉了，但这是 feature，不是 bug。这里举一个需要选择性的任务：给定一个序列，前面的字符都是英文字符，最后以三个下划线结束（例如 "abcdefg\_\_\_"）；要求模型每次读入一个字符，在读入英文字符时输出下划线，遇到下划线后输出它遇到的前三个字符（对上面的例子，输出应该是 "\_\_\_\_\_\_\_abc"）。显然，为了完成这个任务，模型需要学会记数（数到 3），只读入前三个英文字符，中间的字符都忽略掉，最终遇到 _ 时再输出它所记住的三个字符。“只读前三个字符”体现的就是选择性。
	- 从信息变形的角度：信息变形体现在，模型状态在跨时间步时不存在非线性变换，而是加性的（在$C_{t-1}\rightarrow C_{t}$没有经过非线性的激活函数）。假如普通 RNN 的状态里存了某个信息，经过多个时间步以后多次非线性变换把信息变得面目全非了，即使这个信息模型仍然记得，但是也读取不出来它当时到底存的是什么了。而引入门机制的 RNN 单元在各个时间步是乘上一些 0/1 掩码再加新信息，没有非线性变换，这就导致网络想记住的内容（对应掩码为 1）过多个时间步记得还是很清楚。具体参见Towser的文章：https://zhuanlan.zhihu.com/p/34490114

## GRU
根据重置门决定前一时刻的隐状态留多少给当前时刻的隐状态
$$\tilde{h}_{t}=\tanh \left(W_{h} \cdot\left[r_t\cdot h_{t-1}, x_{t}\right]+b_{h}\right)$$
$$r_{t}=\sigma\left(W_{r} \cdot\left[h_{t-1}, x_{t}\right]+b_{r}\right)$$
根据更新门决定要输出多少当前时刻的隐状态和前一时刻的隐状态
$$h_{t}=u_t\cdot\tilde{h}_{t}+\left(1-u_t\right)\cdot h_{t-1}$$
$$u_{t}=\sigma\left(W_{u} \cdot\left[h_{t-1}, x_{t}\right]+b_{u}\right)$$
当$u_t=1,r_t=1$时，GRU退化为普通的RNN
## Batch Normalization：
- 诞生背景：在深层神经网络中，中间某一层的输入是其前一层神经层的输出。因此，其前一层神经层的参数变化会导致当前层输入的分布发生变化。在使用随机梯度下降来训练网络时，每次参数更新都会导致网络中间每一层的输入的分布发生改变，进而每次更新某一层网络的参数都会导致牵一发而动后半身。从机器学习角度来看，如果某个神经层的输入分布发生了改变，那么其参数需要重新学习，这种现象叫做内部协变量偏移（Internal Covariate Shift），其衍生自迁移学习领域的协变量偏移，它是指源空间（训练集）和目标空间（测试集）的条件概率是一致的，但是其边缘概率不同，即：对所有$\boldsymbol{x}\in\mathcal{X},P_{source}(y|\boldsymbol{x})=P_{target}(y|\boldsymbol{x})$，但是$P_{source}(\boldsymbol{x})\neq P_{target}(\boldsymbol{x})$，具体定义参见邱锡鹏老师的《神经网络与深度学习》10.4.2。所以通常协变量偏移都是指的输入层，而这里说的内部协变量偏移则指的是神经网络内部，原因是神经网络每一层的输出正好就是下一层的输入。为了解决内部协变量偏移问题，就要使得每一个神经层的输入的分布在训练过程中要保持稳定（我猜测这里之所以要尽量保持每一层的分布都固定是因为，咱们现在的假设都是神经网络每一层最终都会稳定提取某类特征，最终必然学习到一个稳定的特征分布，不然没法训了。例如：一个分类鸟的模型，我们希望第一层是用来提取嘴的特征，第二层是用来提取嘴形状的特征，那么在我们不断的给模型喂入鸟的图片过程中，我们希望第一层就专注于去学习嘴分布，而第二层就去学嘴形状的分布，而不是在不停地变动）。常见的操作是进行白化（whitening，让均值为0方差为1，https://blog.csdn.net/hjimce/article/details/50864602 和 https://blog.csdn.net/Daniel_djf/article/details/42147109 ），图像领域有尝试过在最开始的输入层对图像进行白化操作，经过白化的图像特征对比度很高，能降低输入特征的冗余，同时也将均值和方差标准化了，但是白化的计算复杂度太高（所以中间网络层没有进行），因此BN提出白化的两种简化方式：1）直接对输入信号的每个维度做规范化；2）在每个mini-batch中计算得到mini-batch mean和variance来替代整体训练集的mean和variance。显然这样做太糙，协变量偏移有很多专门研究的解决办法，比如邱锡鹏老师的《神经网络与深度学习》10.4.2讲的我们可以尝试学习领域无关的特征分布或者调整权重（https://blog.csdn.net/mao_xiao_feng/article/details/54317852 ），而且BN这种归一化方法并不能保证是同样的分布（比如拉普拉斯和正态），具体参见：https://www.zhihu.com/question/38102762/answer/85238569 
- 公式：假设第$l$层神经元的输入为$\boldsymbol{a}^{(l-1)}$，净输入为$\boldsymbol{z}^{(l)}$，输出为$\boldsymbol{a}^{(l)}$
$$\boldsymbol{a}^{(l)}=f\left(\boldsymbol{z}^{(l)}\right)=f\left(W \boldsymbol{a}^{(l-1)}+\boldsymbol{b}\right)$$
$$\begin{aligned} \hat{\boldsymbol{z}}^{(l)} &=\frac{\boldsymbol{z}^{(l)}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \cdot \gamma+\beta \\ & \triangleq \mathrm{BN}_{\gamma, \beta}\left(\boldsymbol{z}^{(l)}\right) \end{aligned}$$
注：
1. BN操作通常都是对净输入为$\boldsymbol{z}^{(l)}$进行操作，虽然也能对$\boldsymbol{a}^{(l-1)}$操作，但是对$\boldsymbol{z}^{(l)}$操作能使得$\boldsymbol{z}^{(l)}$在输入激活函数前达到均值为0方差为1的状态，进而就能有效减缓梯度消失和爆炸，当然这是对Sigmoid和Tanh这类函数来说的，对于ReLU这类激活函数来说有实验表明BN层放在激活函数后面效果更好，因为ReLU的非饱和区只在大于0的区域，由于没有统一定论，最好还是以具体实验效果为主；
2. 进行BN操作相当于加了一个特殊的网络层，所以该层的参数$\gamma,\beta$（$\epsilon$不算参数，仅是用来平滑的固定常量）也参与训练和学习；
3. 其中$\gamma,\beta$是用来缩放和平移的参数，因为如果对净输入$\boldsymbol{z}^{(l)}$进行归一化会使得其取值集中到0附近，如果使用 sigmoid 型激活函数时，这个取值区间刚好是接近线性变换的区间，减弱了神经网络的非线性性质，因此，为了使得归一化不对网络的表示能力造成负面影响，可以通过一个附加的缩放和平移变换改变取值区间（顺便也能照顾一下ReLU这种不是以0为中心的激活函数，不过有实验表明对于ReLU来说通常BN层放在其后面效果更好）。还有一种解释是增加网络的包容能力，因为BN会改变某层原来的输入（因为均值方差标准化本身就会改变原始分布），当然也很有可能不需要改变，不改变的时候就是“还原原来输入”（当$\gamma=\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon},\beta=\mu_{\mathcal{B}}$时）。如此一来，既可以改变同时也可以保持原输入，那么模型的包容能力就提升了，也不高度依赖参数初始化了。
4. 训练时是按batch训练，但是预测时只有一个样本，无法按batch计算均值和方差。此时每个BN层的均值和方差可以考虑将全体训练样本输入最终模型然后计算每一层的均值和方差作为最终模型的参数，当然这种效果虽好但是计算量大（不然的话就不会训练时采用按batch来normalization），所以不太可行。通常实践中的做法是：对每一个BN层在每个batch训练时保存计算出来的均值和方差，最后把所有batch保存下来的均值和方差算加权平均作为最终模型该BN层的均值和方差。
5. 由于要按Batch来求均值和方差，所以训练的时候Batchsize不能太小，不然均值和方差不具有代表性，但是实验表明Batchsize又不能太大，太大训练出来的模型泛化能力差，来源参见：https://zhuanlan.zhihu.com/p/43200897
6. BN不适合用在RNN这种动态网络结构中，因为文本长度不定，尤其对于特别长的文本，其尾部的均值和方差计算有严重偏差，例如对以下含有两个句子的Batch算均值和方差
$$\begin{aligned}
&L1:[[0.1,0.3,0.6][0.3,0.7,0.8]] \\
&L2:[[0.2,0.4,0.7]]
\end{aligned}$$
显然，由于L1的长度为2，L2的长度为1，所以在第二个时间步上就已经没法算出准确的均值和方差了。所以在RNN这种动态网络结构中通常采用LayerNormalization（LN）。有实验表明，在用LN的时候去掉$\gamma,\beta$效果更好，具体参见：https://tobiaslee.top/2019/11/21/understanding-layernorm/
- BN为何有效？
虽说原论文说是因为解决了ICS而有效的，但是后来证明BN并未解决ICS的问题，因为后来有实验证明（MIT2018的《How Does Batch Normalization Help Optimization?》）：即使是应用了BN，网络隐层中的输出仍然存在严重的ICS问题；另一方面也证明了：在BN层输出后人工加入噪音模拟ICS现象，并不妨碍BN的优秀表现。这两方面的证据互相佐证来看的话，其实侧面说明了BN和ICS问题并没什么关系。而且，还有实验表明ICS问题在较深的网络中确实是普遍存在的，但是这并非导致深层网络难以训练的根本原因。那么BN有效的真正原因到底是什么呢？主要原因是加入BN以后损失函数的L-Lipschitz常数更小更稳定，也即损失函数本身更为平滑，所以更易达到最优解，具体参见：https://zhuanlan.zhihu.com/p/43200897
Lipschitz条件，即利普希茨连续条件（Lipschitz continuity）。其定义为：对于函数$f(\boldsymbol x)$，若其定义域中任意的$\boldsymbol x_1,\boldsymbol x_2$，都存在$L>0$，使得
$$|f(\boldsymbol x_1)-f(\boldsymbol x_2)|\leq L\|\boldsymbol x_1-\boldsymbol x_2\|$$
大白话就是：存在一个实数$L$，使得对于函数$f(\boldsymbol x)$上的每对点，连接它们的线的斜率的绝对值不大于这个实数$L$（也即为斜率的上确界），我们称$L$为Lipschitz常数。那么为什么会使得损失函数变平滑呢？其原因很可能是因为加入BN以后，给权重带来了一种“伸缩不变性”，也就是使得回传的梯度不受权重伸缩的影响，那么这样一来权重就不会肆意伸缩从而导致损失函数崎岖，具体参见：
https://www.zhihu.com/question/38102762/answer/391649040
- LN为何有效？
LayerNorm 起作用的原因：一方面通过使得前向传播的输入分布变得稳定；另外一方面，使得后向的梯度更加稳定。二者相比，梯度带来的效果更加明显一些。具体参见：https://tobiaslee.top/2019/11/21/understanding-layernorm/


## Dropout
- 方法：每次（一个batch）训练时随机地忽略部分（通常是一半）神经元，让它们既不参加前向传播也不参加反向梯度更新
- 作用：解决过拟合
- 为什么能解决过拟合？：简单来说，Dropout 通过参数共享提供了一种廉价的 Bagging 集成近似，Dropout 策略相当于集成了包括所有从基础网络除去部分单元后形成的子网络。
	- 取平均的作用： 先回到正常的模型（没有dropout），我们用相同的训练数据去训练5个不同的神经网络，一般会得到5个不同的结果，此时我们可以采用 “5个结果取均值”或者“多数取胜的投票策略”去决定最终结果（Bagging集成学习）。这种“综合起来取平均”的策略通常可以有效防止过拟合问题。因为不同的网络可能产生不同的过拟合，取平均则有可能让一些“相反的”拟合互相抵消。dropout 掉不同的隐藏神经元就类似在训练不同的网络（随机删掉一半隐藏神经元导致网络结构已经不同)，整个dropout过程就相当于 对很多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。
	- 减少神经元之间的依赖关系： 因为dropout程序导致两个神经元不一定每次都在一个dropout网络中出现。（这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况）。 迫使网络去学习更加鲁棒的特征 （这些特征在其它的神经元的随机子集中也存在）。换句话说，假如我们的神经网络是在做出某种预测，它不应该对一些特定的线索片段太过敏感，即使丢失特定的线索，它也应该可以从众多其它线索中学习一些共同的模式（鲁棒性）。（这个角度看 dropout就有点像L1，L2正则，减少权重使得网络对丢失特定神经元连接的鲁棒性提高）
- Drpout 与 Bagging 有何不同？
	- 在 Bagging 的情况下，所有模型都是独立的；而在 Dropout 的情况下，所有模型共享参数，其中每个模型继承父神经网络参数的不同子集。
	- 在 Bagging 的情况下，每一个模型都会在其相应训练集上训练到收敛。而在 Dropout 的情况下，通常大部分模型都没有显式地被训练；取而代之的是，在单个步骤中我们训练一小部分的子网络，参数共享会使得剩余的子网络也能有好的参数设定。
- 缺点：训练时间通常会长一些；
- 与BN的关系：一起使用的话通常会冲突导致性能不佳，非要用的话就只能在BN层后面使用Dropout，或者修改Dropout公式（如使用高斯Dropout）使得它对方差不是那么敏感，其主要原因还是ICS的问题；
- 注意：模型预测阶段需要关闭Dropout，Pytorch中的model.train(), model.eval()能分别自动在训练阶段和预测阶段开启和关掉Dropout
## 语言模型
- 语言模型的本质：计算一串字符串作为一个句子出现的概率（无论是否符合语法）
- 建模思路：理想的模型是对（贝叶斯公式分解）
$$P(s)=P(w_1,w_2,...,w_l)=\prod_{i=1}^l P(w_i|w_1...w_{i-1})$$
进行建模，显然参数量随着字典大小呈指数型增长。因此需要引入假设
$$P(w_i|w_1...w_{i-1})=P(w_i|Context(w_i))$$
- n-gram：直接统计计算$P(w_i|w_1...w_{i-1})=P(w_i|Context(w_i))$
	- uni-gram：
$$P(s)=\prod_{i=1}^l P(w_i|w_1...w_{i-1})\approx \prod_{i=1}^l P(w_i)$$
	- bi-gram（一阶马尔科夫链）：
$$P(s)=\prod_{i=1}^l P(w_i|w_1...w_{i-1})\approx \prod_{i=1}^l P(w_i|w_{i-1})$$
	- tri-gram（二阶马尔科夫链）：
$$P(s)=\prod_{i=1}^l P(w_i|w_1...w_{i-1})\approx \prod_{i=1}^l P(w_i|w_{i-1},w_{i-2})$$
	- 缺点：参数就是各种上下文组合的频率，所以参数量极大；参数量随着n的增大呈指数级增长（通常n超过3效果提升就不大了）；对于未见过的上下文需要使用平滑技术；
- 机器学习：通过最大化似然函数
$$L(\theta)=\prod_{i=1}^lP(w_i|Context(w_i);\theta)$$
学得一个模型，然后用模型给出$P(w_i|Context(w_i))$，相比于n-gram的优势在于参数量大大减少，而且通常还不用考虑平滑。
## word2vec
- 计算过程：https://www.zhihu.com/question/44832436/answer/266068967
- CBOW：用周围词预测中心词
- Skip-Gram：用中心词预测周围词
- CBOW/Skip-Gram:
	- CBOW的训练次数比Skip-Gram的训练次数少
	- Skip-Gram比CBOW训练出来的词向量质量更高，因为在skip-gram当中，每个词都要收到周围的词的影响，每个词在作为中心词的时候，都要进行K次的预测、调整。因此， 当数据量较少，或者词为生僻词出现次数较少时， 这种多次的调整会使得词向量相对的更加准确。因为尽管cbow从另外一个角度来说，某个词也是会受到多次周围词的影响（多次将其包含在内的窗口移动），进行词向量的跳帧，但是他的调整是跟周围的词一起调整的，grad的值会平均分到该词上， 相当于该生僻词没有受到专门的训练，它只是沾了周围词的光而已；
- 目标函数：最大化一个句子的概率
$$L(\theta)=\prod_{w}P(w|Context(w);\theta)$$
- Hierarchical Softmax和Negative Sampling：本质都是对$P(w|Context(w);\theta)$里的softmax进行近似计算，将多分类近似转为二分类计算。为什么会采用这种近似？主要是为了加速训练，而且word2vec的最终目标是词向量，loss只要不是太高就好。
- Hierarchical Softmax【将一次多分类转化为多次二分类】：实质上生成一颗带权路径最短的哈夫曼树，让高频词搜索路径变短；
$$P(w|Context(w);\theta)=\prod_{j=1}^{l^{w}}\left[\sigma\left(\boldsymbol{x}_{w}^{T} \theta_{j}^{w}\right)\right]^{d_{j}^{w}} \cdot\left[1-\sigma\left(\boldsymbol{x}_{w}^{T} \theta_{j}^{w}\right)\right]^{1-d_{j}^{w}}$$
$$P(Context(w)|w;\theta)=\prod_{u\in Context(w)}\prod_{j=1}^{l^{u}}\left[\sigma\left(\boldsymbol{v}_{w}^{T} \theta_{j}^{u}\right)\right]^{d_{j}^{u}} \cdot\left[1-\sigma\left(\boldsymbol{v}_{w}^{T} \theta_{j}^{u}\right)\right]^{1-d_{j}^{u}}$$
上式可以看做在最大化上下文词向量$\boldsymbol{x}_{w}=\frac{\sum_{c\in Context(w)}\boldsymbol{v}_c}{|Context(w)|}$与中心词$w$所相连的那条路径的概率，而路径长度大约等于一颗完全二叉树的高度$\log_2|V|$，相比于softmax每次都要算$|V|$次，效率提高了很多，具体参见：https://blog.csdn.net/itplus/article/details/37969979
- Negative Sampling【将一次多分类转化为一次二分类】：比层次softmax更为直接，实质上对每一个样本中每一个词都进行负例采样；
$$P(w|Context(w);\theta)=\prod_{u\in\{w\}\cup NEG(w)}\left[\sigma\left(\boldsymbol{x}_{w}^{T} \theta^{w}\right)\right]^{\mathbb{I}(u=w)}\left[1-\sigma\left(\boldsymbol{x}_{w}^{T} \theta^{u}\right)\right]^{\mathbb{I}(u\neq w)}$$
$$P(Context(w)|w;\theta)=\prod_{\boldsymbol{u}\in Context(w)}\prod_{z\in\{u\}\cup NEG(u)}\left[\sigma\left(\boldsymbol{v}_{w}^{T} \theta^{u}\right)\right]^{\mathbb{I}(z=u)}\left[1-\sigma\left(\boldsymbol{v}_{w}^{T} \theta^{z}\right)\right]^{\mathbb{I}(z\neq u)}$$
上式可以看做在最大化上下文词向量$\boldsymbol{x}_{w}$与中心词$w$的概率的同时还在最小化$\boldsymbol{x}_{w}$与$w$的负样本$u$的概率，具体参见：https://blog.csdn.net/itplus/article/details/37998797
- Negative Sampling（NEG）和 Noise Contrastive Estimation（NCE）？
NEG是NCE（它本身也是softmax的渐进近似）的特例，NCE不光考虑负样本还考虑负样本的概率分布，而NEG不考虑负样本的分布，或者说NEG默认的概率分布为均匀分布。具体参见：https://www.zhihu.com/question/321088108/answer/659611684
- word2vec和NNLM对比有什么区别？
	- 其建模思想都是基于语言模型；
	- 词向量只不过NNLM一个副产物，word2vec虽然其本质也是语言模型，但是其专注于词向量本身，因此做了许多优化来提高计算效率（比如词向量直接sum，不再拼接，并舍弃隐层）；
	- word2vec可以看成是一个单隐层的线性神经网络，之所以说是线性是由于它的隐层并没有使用激活函数（或者说直接舍弃隐层），而之所以使用线性模型是因为我们希望捕捉到一组具有线性关系的词向量的表示（比如king - man = queen - woman）.
- 实际训练出了两套词向量，用哪个？：如果是纯Softmax训练的话确实是有两组词向量，两者都有上下文语义，所以都可以用，不过通常都用第一组或者两组加和取平均（有实验表明后者效果更好）；如果是采用哈夫曼这种softmax近似算法的话就只能用第一组了，因为对于哈夫曼来说，第二组的辅助向量个数通常没有字典那么多，无法一一对应，而且哈夫曼的不靠近叶结点的辅助向量没有语义（负采样应该没问题，因为每个词都会做一次中心词，所以每个词都有对应的辅助向量，且有语义）；具体参见：https://www.zhihu.com/question/278791534
## Transformer
- 常见的Attention机制都有哪些？
$$\alpha_i=P(z=i|\mathbf{X},\boldsymbol q)=softmax\left(s(\boldsymbol x_i,\boldsymbol q)\right)$$
其中$\alpha_i$称为注意力分布，$s(\boldsymbol x_i,\boldsymbol q)$为注意力打分函数，常见的打分函数计算方法如下：
	- 加性模型：$s(\boldsymbol x_i,\boldsymbol q)=\boldsymbol v^T tanh(W\boldsymbol x_i+U\boldsymbol q)$
	- 点积模型：$s(\boldsymbol x_i,\boldsymbol q)=\boldsymbol x_i^T\boldsymbol q$
	- 缩放点积模型：$s(\boldsymbol x_i,\boldsymbol q)=\frac{\boldsymbol x_i^T\boldsymbol q}{\sqrt d}$
	- 双线性模型：$s(\boldsymbol x_i,\boldsymbol q)=\boldsymbol x_i^TW\boldsymbol q$ 
其中，$W,U,\boldsymbol v$都是可学习的参数，$d$为输入向量的维度。理论上，加性模型和点积模型的复杂度差不多，但是点积模型在实现上可以更好地利用矩阵乘积，从而计算效率更高。缩放点积模型是为了改进点积模型值太大的问题；双线性模型是为了在计算相似度时引入非对称性。
- 为什么采用点积模型的Attention而不采用加性模型？
主要原因是在理论上，加性模型和点积模型的复杂度差不多，但是点积模型在实现上可以更好地利用矩阵乘法，而矩阵乘法有很多加速策略，因此能加速训练。但是论文中实验表明，当维度$d$越来越大时，加性模型的效果会略优于点积模型，原因应该是加性模型整体上还是比点积模型更复杂（有非线性因素）。
- attention为什么scaled？为什么除以$\sqrt{d}$呢？
	- 因为对于$(\boldsymbol q_i^T\boldsymbol k_1,\boldsymbol q_i^T\boldsymbol k_2,...,\boldsymbol q_i^T\boldsymbol k_n)$来说，如果某个$\boldsymbol q_i^T\boldsymbol k_j$相对于其他元素很大的话，那么对此向量softmax后就容易得到一个onehot向量，不够“soft”了，而且反向传播时梯度为0会导致梯度消失；
$$\boldsymbol y=softmax(\boldsymbol x)$$
$$\frac{\partial \boldsymbol y}{\partial \boldsymbol x}=diag(\boldsymbol y)-\boldsymbol y\boldsymbol y^T=\left[\begin{array}{cccc}{\frac{\partial y_{1}}{\partial x_{1}}} & {\frac{\partial y_{2}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{m}}{\partial x_{1}}} \\ {\frac{\partial y_{1}}{\partial x_{2}}} & {\frac{\partial y_{2}}{\partial x_{2}}} & {\cdots} & {\frac{\partial y_{m}}{\partial x_{2}}} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial y_{1}}{\partial x_{n}}} & {\frac{\partial y_{2}}{\partial x_{n}}} & {\cdots} & {\frac{\partial y_{m}}{\partial x_{n}}}\end{array}\right]$$
	- 原论文是这么解释的：假设每个$\boldsymbol q\in R^d$和$\boldsymbol k\in R^d$的每个维度都是服从均值为0方差为1的，那么二者的内积$\boldsymbol q^T\boldsymbol k$的均值就是0，方差就是$d$，所以内积的方差和原始方差之间的比例大约是维度值$d$，为了降低内积各个维度值的方差（这样各个维度取值就在均值附近，不会存在某个维度偏离均值太远），所以要除以$\sqrt{d}$（标准差），具体参见：https://www.zhihu.com/question/339723385/answer/782509914
- 为什么需要位置向量？
因为self-attention的结构本身就直接忽略了词序信息，使得同一个词无论在哪个位置它的qkv都是完全一样的，但是词序信息在某些任务场景下是很重要的特征，比如情感分析：“我喜欢这部电影因为它不低俗”和“我不喜欢这部电影因为它低俗”，这两句话中“不”字位置的不同直接导致两句话所表述的情感不同，因此需要通过位置来弥补self-Attention的缺陷；
- Position encoding：
$$PE(pos,2i)=\sin\left(\cfrac{pos}{10000^{2i/d}}\right)$$
$$PE(pos,2i+1)=\cos\left(\cfrac{pos}{10000^{2i/d}}\right)$$
其中，$pos$代表的即为position，设句子长度为$L$，则$pos=0,1,2,...,L-1$，$d$为向量的维度，而$i$则为具体的某一维度，设PE的维度$d=512$，那么$i=0,1,2,...,255$。Position encoding采用这种编码方式是具备相对位置信息的，原因如下：
$$\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$$
$$\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta$$
所以
$$PE(pos+k,2i)=PE(pos,2i)\times PE(k,2i+1)+PE(pos,2i+1)\times PE(k,2i)$$
$$PE(pos+k,2i+1)=PE(pos,2i+1)\times PE(k,2i+1)-PE(pos,2i)\times PE(k,2i)$$
显然，上式表示pos+k的位置编码是pos和k的线性组合，那么也就意味着蕴含相对位置信息，比如5=3+2，6=4+2。
- Position encoding为什么选择相加而不是拼接呢？
因为$[W1 W2][e; p] = W1e + W2p,W(e+p)=We+Wp$，就是说求和相当于拼接的两个权重矩阵共享（W1=W2=W），但是这样权重共享是明显限制了表达能力的。
- position encoding和position embedding的区别？
P-enc构造简单直接无需额外的学习参数；能兼容预训练阶段的最大文本长度和训练阶段的最大文本长度不一致；
P-emb构造也简单直接但是需要额外的学习参数；训练阶段的最大文本长度不能超过预训练阶段的最大文本长度（因为没学过这么长的，不知道如何表示）；但是P-emb的潜力在直觉上会比P-enc大，因为毕竟是自己学出来的，只有自己才知道自己想要什么（前提是数据量得足够大）。
- 为何17年提出Transformer时采用的是P-enc而不是P-emb？而Bert却采用的是P-emb？
Transformer 的作者在论文中对比了 Position Encoder 和 Position Embedding,在模型精度上没有明显区别。出于对序列长度限制和参数量规模的考虑,最终选择了 Encode 的形式。那么为什么Bert不这么干呢？主要原因如下：
	- 模型的结构需要服务于模型的目标：Transformer最开始提出是针对机器翻译任务的，而机器翻译任务对词序特征要求不高，因此在效果差不多的情况下选择P-enc足矣。但是Bert是作为通用的预训练模型，下游任务有很多对词序特征要求很高，因此选择潜力比较大的P-emb会更好；
	- 数据量的角度：Transformer用的数据量没有Bert的数据量大，所以使用潜力无限的P-emb会有大力出奇迹的效果；
- Feed-Forward Networks（FFN，前馈神经网络）起什么作用？
FFN在每个位置参数共享，但是不同层之间参数不共享。由于self-Attention主要都是线性操作，而FFN是Transformer里唯一提供非线性变换的结构，所以FFN的主要作用是进行非线性变换。具体参见：https://zhuanlan.zhihu.com/p/60821628
- 为什么用残差结构？
防止网络叠得太深时出现网络退化，原因参见：https://zhuanlan.zhihu.com/p/80226180
从反向传播的角度：在前向传播时，输入信号可以从任意低层直接传播到高层；反向传播时，第$t$层的损失也能直接传播到第$1$层
$$\begin{aligned}
h_t&=h_{t-1}+f(h_{t-1}) \\
&=h_{t-2}+f(h_{t-2})+f(h_{t-1}) \\
&=h_{1}+\sum_{i=1}^{t-1}f(h_{i}) \\
\end{aligned}$$
$$\begin{aligned}
\frac{\partial L}{\partial h_1}&=\frac{\partial L}{\partial h_t}\frac{\partial h_t}{\partial h_{1}} \\
&=\frac{\partial L}{\partial h_t}\left(1+\frac{\partial }{\partial h_{1}}\sum_{i=1}^{t-1}f(h_{i})\right) \\
&=\frac{\partial L}{\partial h_t}+\frac{\partial L}{\partial h_t}\frac{\partial }{\partial h_{1}}\sum_{i=1}^{t-1}f(h_{i}) \\
\end{aligned}$$
从集成学习的角度：残差结构就像并行训练了多个网络模型然后集成在一起
- Label Smoothing
在常见的多分类问题中，先经过softmax处理后进行交叉熵计算，原理很简单可以将计算loss理解为，为了使得网络对测试集预测的概率分布和其真实分布接近，常用的做法是使用one-hot对真实标签进行硬编码，作者认为这种将标签硬编码的方式会使得网络过于自信而导致过拟合，因此可以考虑软化这种编码方式。
$$p^{\prime}(k)=(1-\epsilon)p(k)+\epsilon u(k)=(1-\epsilon)p(k)+\frac{\epsilon}{K}$$
其中$u(k)$是关于类别$k$的均匀分布，$K$为总类别数，$p(k)$为硬编码的分布。当$k$等于其真实标记时，$p(k)=1$，而
$$p^{\prime}(k)=(1-\epsilon)+\frac{\epsilon}{K}$$
当$k$不等于其真实标记时，$p(k)=0$，而
$$p^{\prime}(k)=\frac{\epsilon}{K}$$
所以，综合看来就是把最高得分砍掉一点，多出来的概率平均分给所有人。与硬编码的交叉熵损失函数类似，只需将硬编码交叉熵损失函数中的$p(x)$换成$p^{\prime}(k)$即可
$$\begin{aligned}
H(p^{\prime},q)&=-\sum_kp^{\prime}(k)\log q(k) \\
&=-\sum_k\left[(1-\epsilon)p(k)+\epsilon u(k)\right]\log q(k) \\
&=-\sum_k(1-\epsilon)p(k)\log q(k)-\sum_k\epsilon u(k)\log q(k) \\
&=(1-\epsilon)H(p,q)-\epsilon H(u,q) \\
\end{aligned}$$
- 缺点：缺点在原文中没有提到，是后来在Universal Transformers（https://zhuanlan.zhihu.com/p/44655133 ）中指出的，在这里加一下吧，主要是两点：
  - 实践上：有些rnn轻易可以解决的问题transformer没做到，比如复制string，或者推理时碰到的sequence长度比训练时更长（因为碰到了没见过的position embedding）
  - 理论上：transformers非computationally universal（图灵完备），（我认为）因为无法实现“while”循环
- q,k,v能合并成一个吗？
也可以，但是得改变Attention的计算方式，至少现在的缩放点积Attention会使得$q_i$和它本身的得分最大，这样会使得当前词的词向量几乎还是它本身，而没法达到self-attention用上下文信息来增强当前词的语义表示的目的。虽然合并以后看起来参数少了，但是在self-attention过程中会出现平方项，优化起来会麻烦许多（w2v也是因此才弄了两个参数矩阵来学习），所以还不如分成3个。具体参见：
https://www.zhihu.com/question/278791534/answer/706466139
- 源码要点：
  - q和k的维度是$d_k$，v的维度是$d_v$，两个维度可以不同？
    - 可以不同，但是实际用的时候预设的是相同的
  - Decoder训练时的Masked和预测时的Masked的解释参见：https://zhuanlan.zhihu.com/p/58009338
  - Dropout都用在哪些地方？
    - 1.word-embedding和position-encoding拼接后有一个；
    - 2.self-attention里面算完得分的softmax以后有一个；
    - 3.LN完会有一个`return x + self.dropout(sublayer(self.norm(x)))`；
    - 4.全连接层那里经过ReLU函数后有一个

## Bert
- Masked LM（借鉴CBOW）
在BERT中, Masked LM(Masked language Model)构建了语言模型, 这也是BERT的预训练中任务之一, 简单来说, 就是随机遮盖或替换一句话里面任意字或词, 然后让模型通过上下文的理解预测那一个被遮盖或替换的部分, 之后做Loss的时候只计算被遮盖部分的Loss, 其实是一个很容易理解的任务, 实际操作方式如下。随机把一句话中15%的token替换成以下内容:
  1) 这些token有80%的几率被替换成[mask];
  2) 有10%的几率被替换成任意一个其他的token;
  3) 有10%的几率原封不动.
- Next Sentence Prediction（借鉴Skip-gram）
	- 首先我们拿到属于上下文的一对句子, 也就是两个句子, 之后我们要在这两段连续的句子里面加一些特殊token：[cls]上一句话[sep]下一句话.[sep]。也就是在句子开头加一个[cls], 在两句话之中和句末加[sep]
	- 我们看到上图中两句话是[cls] my dog is cute [sep] he likes playing [sep], [cls]我的狗很可爱[sep]他喜欢玩耍[sep], 除此之外, 我们还要准备同样格式的两句话, 但他们不属于上下文关系的情况;
[cls]我的狗很可爱[sep]企鹅不擅长飞行[sep], 可见这属于上下句不属于上下文关系的情况;
在实际的训练中, 我们让上面两种情况出现的比例为1:1, 也就是一半的时间输出的文本属于上下文关系, 一半时间不是;
	- 我们进行完上述步骤之后, 还要随机初始化一个可训练的segment embeddings, 见上图中, 作用就是用embeddings的信息让模型分开上下句, 我们给上句全0的token, 下句给全1的token, 让模型得以判断上下句的起止位置, 例如:
[cls]我的狗很可爱[sep]企鹅不擅长飞行[sep]
0 0  0  0  0  0  0  0   1  1  1  1  1  1  1  1
上面0和1就是segment embeddings.
- Bert为什么好？：除了模型本身提取特征的能力强以外，它还提供了一套通用任务框架，使得BERT能够支持包括：句子对分类任务、单句子分类任务、阅读理解任务和序列标注任务；
- 为什么要mask？：为了解决双向机制在训练语言模型时存在的泄密问题；
- 为什么10%替换成其他token？：因为模型不知道哪些词是被mask的，哪些词是mask了之后又被替换成了一个其他的词，这会迫使模型尽量在每一个词上都学习到一个全局语境下的表征，因而也能够让BERT获得更好的语境相关的词向量（这正是解决一词多义的最重要特性）。其实这样做的更感性解释是，因为模型不知道哪里有坑，所以随时都要提心吊胆，保持高度的警惕。正如一马平川的大西北高速公路，通常认为都是路线直，路面状况好，但如果掉以轻心，一旦有了突发情况，往往也最容易出事故，鲁棒性不高；而反倒是山间小路，明确告诉了每一位司机路面随时都有坑，并且没法老远就提前知道，所以即便老司机也只能小心翼翼的把稳方向盘慢慢的开，这样做反倒鲁棒性更高。
- 为什么10%完全保留？：因为fine-tune的时候输入中是没有`[MASK]`，也没有随机替换的标记的，那么fine-tune阶段看到一个完整的句子时，Bert可能会有点“不知所措”（其实论文最后的实验证明还好，性能损失不大），所以这样存在一个pre-train和fine-tune阶段存在mismatch，所以需要一部分词既不mask也不随机替换。其他人理解是：因为输入层是待预测词的真实embedding，在输出层中的该词位置得到的embedding，是经过层层Self-attention后得到的，这部分embedding里肯定多多少少依然保留有部分输入embedding的信息，而这部分的多多少少就是通过输入一定比例的真实词所带来的额外奖励，最终会使得模型的输出向量朝输入层的真实embedding有一个偏移，而如果全用mask的话，模型只需要保证输出层的分类准确，对于输出层的向量表征并不关心，因此可能会导致最终的向量输出效果并不好，具体参见：https://zhuanlan.zhihu.com/p/50443871
- 为什么有Next Sentence Prediction？：主要是为了学习句子级别的特征，可以看作是句子级别的MASK操作，为下游句子级别的NLP类任务提供支持。但是最近 RoBERTa 又给了 NSP 一锤，说去掉 NSP 更好。他们猜测 BERT 在做 NSP 的隔离实验的时候，可能是只去掉了 NSP loss，但是没改训练数据的生成过程（训练数据中还有两个 segment 不相邻的情况）。考虑到 RoBERTa 就是 BERT，仅仅是改了一些训练超参数的设置，所以基本上给 NSP 判死刑了。详见：https://www.zhihu.com/question/331076024/answer/767060671
## XLNet
- 提出了自回归语言模型（Autoregressive LM）和自编码语言模型（Autoencoder LM）这两个新概念，两者区别主要是前者是指单向训练的语言模型（Elmo和GPT），后者是指双向训练的语言模型（Bert），前者适合做生成类的任务，后者则适合做非生成类的任务，具体详见：https://zhuanlan.zhihu.com/p/70257427
- 指出了Bert的缺点：XLNet在文中指出的，第一个预训练阶段因为采取引入[Mask]标记来Mask掉部分单词的训练模式，而Fine-tuning阶段是看不到这种被强行加入的Mask标记的，所以两个阶段存在使用模式不一致的情形，这可能会带来一定的性能损失；另外一个是，Bert在第一个预训练阶段，假设句子中多个单词被Mask掉，这些被Mask掉的单词之间没有任何关系，是条件独立的，而有时候这些单词之间是有关系的，XLNet则考虑了这种关系
- 提出了Permutation Language Model来改进Bert的第一个问题；
## 词向量演进历史
- one-hot：维度灾难（字层面肯定没问题，关键是词的组合是无穷的）；两两正交毫无语义；
- 基于矩阵分解：
	- LSA/LSI（潜在语义分析/索引）：基于词-文档共现矩阵（也可以用词-词共现矩阵）的SVD，矩阵填充用单词频数或者$TF-IDF=\frac{tf_{ij}}{tf_j}\log\frac{df}{df_i}$，然后假设有k个主题，则矩阵分解结果为$U_k\Sigma_kV^T$，其中$U_k$可以作为词向量矩阵，$\Sigma_kV^T$为文档向量矩阵
		- 优点：利用了全局语料
		- 缺点：词-文档共现矩阵的构建存在维度灾难（矩阵分解得到的词向量不存在维度灾难，因为是降维分解）；SVD计算量大；训练出来的词向量语义差（相比于神经网络建模能力较差），不过还是有一定的可解释性，具体参见：https://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html ；
	- Glove：构建词-词共现矩阵，定义一个奇特的损失函数，详见：https://www.fanyeong.com/2018/02/19/glove-in-detail/
		- 优点：词-词共现矩阵的构建不存在维度灾难；不再使用SVD，而是使用一种高效可并行的损失函数；利用了全局语料；考虑了单词的统计权重（LSA没有考虑）；
		- 缺点：训练出来的词向量语义差（但是相比于神经网络建模能力还是差很多）；
- 基于语言模型：
	- word2vec：起源于NNLM，不过最后有论文证明基于词-词共现矩阵分解的模型与Skip-gram模型有同样的最优解。不过这两个模型只是矩阵和神经网络这两种技术手段下的特例，它们都选用了词作为上下文。
      - 优点：没有维度灾难；神经网络能很灵活地利用各种手段建模上下文；
      - 缺点：没有利用到全局语料；单词表征唯一，无法解决一词多义的问题；
    - elmo：分别学习单词的上文和下文，然后拼接上下文特征向量
      - 优点：具有w2v的所有优点，同时还解决了一词多义问题；
      - 缺点：相比于bert，lstm的特征抽取能力偏弱，且采用拼接的方式进行特征融合效果一般；
- 思路历程
	- 分布假说(distributional hypothesis):上下文相似的词，其语义也相似。该假说由 Harris 在 1954 年提出，并由 Firth 在 1957 年进一步明确和完善。
	- 分布式表示(distributional representation):分布(distributional)描述的是上下文的概率分布，因此用上下文描述语义的表示方法(基于分布假说的方法)都可以称作分布式表示。与之相对的是形式语义表示。
	- 分散式表示(distributed representation):分散(distributed)描述的是把信息分散式地存储在向量的各个维度中，与之相对的是局部表示(local representation)，如词的独热表示(one-hot representation)，在高维向量中每一维都表示文本的某种潜在的语法或语义特征。一般来说，通过矩阵降维或神经网络降维可以将语义分散存储到向量的各个维度中，因此，这类方法得到的低维向量一般都可以称作分散式表示。
	- 总结：分散式表示只要在低维的空间中能够区分出两个词的不同就够了。不一定非得要求意义相近的词距离也相近。因为上层的神经网络可以具有高度非线性，完全可以将原始的表示空间高度扭曲。所以，从Distributed Representation本身来讲，我们不要给它附加太多的意义，比如要满足man−woman ≈ king – queen之类的。Distributed Representation就是将文本分散在低维空间中的很多点上，只要这些点有一定区分性就够了。说得极端一点，我们的词向量都是随机产生的稠密向量，这也是一种distributed表示，不需要有语义上解释（比如分布假说）。只要后续的模型足够强大，一样可以做的很好。总之，Distributional Representation指的是一类获取文本表示的方法，而Distributed Representation指的是文本表示的形式，就是低维、稠密的连续向量。但这两个并不对立。比如Skip-Gram、CBOW和glove等模型得到词向量，即是Distributional Representation，又是Distributed Representation。具体参见：https://zhuanlan.zhihu.com/p/22386230
- 为什么采用语言模型来训练词向量？
要介绍词向量是怎么训练得到的，就不得不提到语言模型。到目前为止我了解到的所有训练方法都是在训练语言模型的同时，顺便得到词向量的。这也比较容易理解，要从一段无标注的自然文本中学习出一些东西，无非就是统计出词频、词的共现、词的搭配之类的信息。而从自然文本中统计并建立一个语言模型，无疑是要求最为精确的一个任务（也不排除以后有人创造出更好更有用的方法）。既然构建语言模型这一任务要求这么高，其中必然也需要对语言进行更精细的统计和分析，同时也会需要更好的模型，更大的数据来支撑。目前最好的词向量都来自于此，也就不难理解了。具体参见：http://licstar.net/archives/328#s1