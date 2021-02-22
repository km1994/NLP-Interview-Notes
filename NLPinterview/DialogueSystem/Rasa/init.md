# 【关于 Rasa 入门】那些你不知道的事

> 作者：杨夕
> 
> 项目地址：https://github.com/km1994/nlp_paper_study
> 
> 个人介绍：大佬们好，我叫杨夕，该项目主要是本人在研读顶会论文和复现经典论文过程中，所见、所思、所想、所闻，可能存在一些理解错误，希望大佬们多多指正。

![](img/微信截图_20210204080923.png)

## 一、创建一个新的Rasa项目

1. 第一步是创建一个新的Rasa项目。要做到这一点，运行下面的代码:
  
```s
    rasa init --no-prompt
```

> 注：rasa init命令创建rasa项目所需的所有文件，并根据一些示例数据训练一个简单的机器人。如果你省略了——no-prompt参数，将会询问你一些关于项目设置的问题。

- 运行过程

```s
    $ rasa init --no-prompt
    >>>
    Welcome to Rasa! 🤖

    To get started quickly, an initial project will be created.
    If you need some help, check out the documentation at https://rasa.com/docs/rasa.

    Created project directory at '/web/workspace/yangkm/python_wp/nlu/DSWp'.
    Finished creating project structure.
    Training an initial model...
    Training Core model...
    Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 3562.34it/s, # trackers=1]
    Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 1523.54it/s, # trackers=5]
    Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 380.28it/s, # trackers=20]
    Processed Story Blocks: 100%|█████████████████████████████████████████████| 5/5 [00:00<00:00, 301.26it/s, # trackers=24]
    Processed trackers: 100%|█████████████████████████████████████████████████| 5/5 [00:00<00:00, 2233.39it/s, # actions=16]
    Processed actions: 16it [00:00, 14986.35it/s, # examples=16]
    Processed trackers: 100%|█████████████████████████████████████████████| 231/231 [00:00<00:00, 899.80it/s, # actions=126]
    Epochs:   0%|                                                                                   | 0/100 [00:00<?, ?it/s]/home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/rasa/utils/tensorflow/model_data.py:386: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    final_data[k].append(np.concatenate(np.array(v)))
    Epochs: 100%|████████████████████████████████████| 100/100 [00:06<00:00, 14.77it/s, t_loss=0.083, loss=0.009, acc=1.000]
    2020-09-17 16:46:48 INFO     rasa.utils.tensorflow.models  - Finished training.
    2020-09-17 16:46:48 INFO     rasa.core.agent  - Persisted model to '/tmp/tmpjkpkgun2/core'
    Core model training completed.
    Training NLU model...
    2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Training data stats:
    2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Number of intent examples: 43 (7 distinct intents)
    2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  -   Found intents: 'mood_unhappy', 'bot_challenge', 'deny', 'affirm', 'greet', 'mood_great', 'goodbye'
    2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Number of response examples: 0 (0 distinct responses)
    2020-09-17 16:46:48 INFO     rasa.nlu.training_data.training_data  - Number of entity examples: 0 (0 distinct entities)
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component WhitespaceTokenizer
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component RegexFeaturizer
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component LexicalSyntacticFeaturizer
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component CountVectorsFeaturizer
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component CountVectorsFeaturizer
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:48 INFO     rasa.nlu.model  - Starting to train component DIETClassifier
    /home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/rasa/utils/common.py:363: UserWarning: You specified 'DIET' to train entities, but no entities are present in the training data. Skip training of entities.
    Epochs:   0%|                                                                                   | 0/100 [00:00<?, ?it/s]/home/amy/.conda/envs/yangkm/lib/python3.6/site-packages/rasa/utils/tensorflow/model_data.py:386: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    final_data[k].append(np.concatenate(np.array(v)))
    Epochs: 100%|████████████████████████████████| 100/100 [00:05<00:00, 18.36it/s, t_loss=1.475, i_loss=0.095, i_acc=1.000]
    2020-09-17 16:46:58 INFO     rasa.utils.tensorflow.models  - Finished training.
    2020-09-17 16:46:59 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:59 INFO     rasa.nlu.model  - Starting to train component EntitySynonymMapper
    2020-09-17 16:46:59 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:59 INFO     rasa.nlu.model  - Starting to train component ResponseSelector
    2020-09-17 16:46:59 INFO     rasa.nlu.selectors.response_selector  - Retrieval intent parameter was left to its default value. This response selector will be trained on training examples combining all retrieval intents.
    2020-09-17 16:46:59 INFO     rasa.nlu.model  - Finished training component.
    2020-09-17 16:46:59 INFO     rasa.nlu.model  - Successfully saved model into '/tmp/tmpjkpkgun2/nlu'
    NLU model training completed.
    Your Rasa model is trained and saved at '/web/workspace/yangkm/python_wp/nlu/DSWp/models/20200917-164632.tar.gz'.
    If you want to speak to the assistant, run 'rasa shell' at any time inside the project directory.
```


## 二、Rasa 目录生成内容介绍

将在该目录下参加以下文件：

<table>
    <thead>
        <td>文件名称</td><td>作用说明</td>
    </thead>
    <tr>
        <td>init.py</td><td>帮助python查找操作的空文件</td>
    </tr>
    <tr>
        <td>actions.py</td><td>为你的自定义操作编写代码</td>
    </tr>
    <tr>
        <td>config.yml ‘*’</td><td>配置NLU和Core模型</td>
    </tr>
    <tr>
        <td>credentials.yml</td><td>连接到其他服务的详细信息</td>
    </tr>
    <tr>
        <td>data/nlu.md ‘*’</td><td>你的NLU训练数据</td>
    </tr>
    <tr>
        <td>data/stories.md ‘*’</td><td>你的故事</td>
    </tr>
    <tr>
        <td>config.yml ‘*’</td><td>配置NLU和Core模型</td>
    </tr>
    <tr>
        <td>domain.yml ‘*’</td><td>你的助手的域</td>
    </tr>
    <tr>
        <td>endpoints.yml</td><td>接到fb messenger等通道的详细信息</td>
    </tr>
    <tr>
        <td>models/.tar.gz</td><td>你的初始模型</td>
    </tr>
</table>

> 注：最重要的文件用“*”标记。你将在本教程中了解所有这些文件。

## 三、Rasa 对话系统测试

```s
    $ rasa shell
    >>>
    2021-01-30 18:01:25.946702: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
    2021-01-30 18:01:25.946959: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2021-01-30 18:01:28.518362: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library nvcuda.dll
    2021-01-30 18:01:28.938568: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties:
    pciBusID: 0000:01:00.0 name: GeForce 940MX computeCapability: 5.0
    coreClock: 0.8605GHz coreCount: 4 deviceMemorySize: 2.00GiB deviceMemoryBandwidth: 37.33GiB/s
    2021-01-30 18:01:28.939909: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
    2021-01-30 18:01:28.940935: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
    2021-01-30 18:01:28.942048: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
    2021-01-30 18:01:28.942970: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
    2021-01-30 18:01:28.943862: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
    2021-01-30 18:01:28.945788: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
    2021-01-30 18:01:28.950147: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudnn64_7.dll'; dlerror: cudnn64_7.dll not found
    2021-01-30 18:01:28.950220: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1753] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...
    2021-01-30 18:01:28 INFO     rasa.model  - Loading model models\20210130-175951.tar.gz...
    2021-01-30 18:01:31 INFO     root  - Connecting to channel 'cmdline' which was specified by the '--connector' argument. Any other channels will be ignored. To connect to all given channels, omit the '--connector' argument.
    2021-01-30 18:01:31 INFO     root  - Starting Rasa server on http://localhost:5005
    2021-01-30 18:01:31 INFO     rasa.model  - Loading model models\20210130-175951.tar.gz...
    2021-01-30 18:01:36.663436: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2021-01-30 18:01:36.676013: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x27e2f29b300 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2021-01-30 18:01:36.676366: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2021-01-30 18:01:36.677138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
    2021-01-30 18:01:36.677640: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]
    2021-01-30 18:01:45 INFO     root  - Rasa server is up and running.
    Bot loaded. Type a message and press enter (use '/stop' to exit):
    Your input ->  hello
    Hey! How are you?
    Your input ->  ok
    Great, carry on!
    Your input ->  yes
    Great, carry on!
```


## 参考资料

1. [Rasa 文档](https://rasa.com/docs/rasa/)
2. [Rasa 安装](http://rasachatbot.com/2_Rasa_Tutorial/#rasa)
3. [Rasa 聊天机器人中文官方文档|磐创AI](http://rasachatbot.com/)
4. [Rasa 学习](https://blog.csdn.net/ljp1919/category_9656007.html)
5. [rasa_chatbot_cn](https://github.com/GaoQ1/rasa_chatbot_cn)
6. [用Rasa NLU构建自己的中文NLU系统](http://www.crownpku.com/2017/07/27/用Rasa_NLU构建自己的中文NLU系统.html)
7. [Rasa_NLU_Chi](https://github.com/crownpku/Rasa_NLU_Chi)
8. [_rasa_chatbot](https://github.com/zqhZY/_rasa_chatbot)
9. [rasa 源码分析](https://www.zhihu.com/people/martis777/posts)