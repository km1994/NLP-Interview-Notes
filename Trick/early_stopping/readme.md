# 【关于 早停法 EarlyStopping 】那些你不知道的事

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

- [【关于 早停法 EarlyStopping 】那些你不知道的事](#关于-早停法-earlystopping-那些你不知道的事)
  - [一、 为什么要用 早停法 EarlyStopping？](#一-为什么要用-早停法-earlystopping)
  - [二、 早停法 EarlyStopping 是什么？](#二-早停法-earlystopping-是什么)
  - [三、早停法 torch 版本怎么实现？](#三早停法-torch-版本怎么实现)

## 一、 为什么要用 早停法 EarlyStopping？

模型训练过程中，训练 loss 和 验证 loss 在训练初期都是 呈下降趋势；当训练到达一定程度之后， 验证 loss 并非继续随 训练 loss 一样下降，而是 出现上升的趋势，此时，如果继续往下训练，容易出现 模型性能下降问题，也就是我们所说的过拟合问题。

那么，有什么办法可以避免模型出现该问题呢？

这个就是本节 所介绍的方法 —— 早停法

## 二、 早停法 EarlyStopping 是什么？

早停法 就是在训练中计算模型在验证集上的表现，当模型在验证集上的表现开始下降的时候，停止训练，这样就能避免模型由于继续训练而导致过拟合的问题。所以说 早停法 结合交叉验证法可以防止模型过拟合。

## 三、早停法 torch 版本怎么实现？

```python
import torch
import numpy as np
# 早停法
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, model_path):
        '''
            功能：早停法 计算函数
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    # 功能：当验证损失减少时保存模型
    def save_checkpoint(self, val_loss, model, model_path):
        '''
            功能：当验证损失减少时保存模型
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint_loss.pt')
        torch.save(model, open(model_path, "wb"))
        self.val_loss_min = val_loss
```
