# 【关于 Rasa 中文 任务型对话系统开发】那些你不知道的事

## 目录

## 一、流程介绍




## 二、 NLU 模块

### 2.1 思路介绍

- 在 data/nlu.md 添加 意图 intent 【以添加 点外卖为例】
  - 确定 意图 名称：request_takeway；
  - 确定需要的 槽位：refreshments、tea、address、phone_number；
  - 确定 用户 query；
  - 确定 当 某个 槽位 为空时，Bot 通过回复 什么获取 对应 槽位值

### 2.2 意图 和 槽位 设置

> 举例

```s
## intent: ask_tea
- 请问，这边有什么茶呢？
...

## intent: ask_refreshments
- 请问，这边有什么好吃的？
...

## intent: ask_address
- 请问送货地址在哪里呢？
...

## intent: ask_phone_number
- 需要手机号么？

## intent: request_takeway
- 点份外卖
- 点一份[肠粉](refreshments)，一杯[茉莉花茶](tea)，送到[南山区](address)，电话号码为[13025240602](phone_number)
- 麻烦送一份[蛋糕](refreshments)到[南山区](address)
- ...
```

## 三、 stories 模块 设置

### 3.1 为什么需要 stories 模块 ？

因为 NLU 识别出 query 的意图和槽位之后，需要 确定 其 所触发的对应事件。

```s
    eg：
        query：  用户：我要点一份外卖
        NLU:     {intent:"订外卖",slots:{}}
        action:  takeway_form
```
