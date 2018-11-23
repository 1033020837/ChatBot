这是一个使用Tensorflow框架通过Sequence To Sequence模型训练出的简单的聊天机器人。  
   
先上一张效果截图：  
  
  
![效果截图](https://github.com/1033020837/ChatBot/blob/master/img/%E8%81%8A%E5%A4%A9%E6%88%AA%E5%9B%BE.png)  
  
  
下面简单的介绍一下这个项目。  

## 1.参考资料  
理论部分：  
 [吴恩达深度学习教程SequenceToSequence部分](https://mooc.study.163.com/learn/2001280005?tid=2001391038#/learn/content?type=detail&id=2001771062&cid=2001777016)  
 API部分：  
 [Tensorflow Python API文档](https://tensorflow.google.cn/api_docs/python/tf)  
 中文语料的获取：  
 [中文开放聊天语料整理](https://github.com/codemayq/chaotbot_corpus_Chinese)    
 
## 2.项目结构介绍  
 - config.py  整个项目的配置文件，如语料库的存放位置，模型的参数等  
 - data_unit.py 处理语料库的类，对原始语料进行清洗，并生成批训练数据。  
 - seq2seq.py 构建了一个Sequence To Sequence模型，包含编码器、解码器、优化器、训练过程、预测过程等部分。
 - train.py 用于模型的训练。  
 - predict.py 用于模型的测试。  
 - data 该文件夹用于保存语料文件。  
 - model 该文件夹用于保存训练好的模型。由于github不能上传超过100M的文件，所以我把模型上传到了百度云上面，地址：[https://pan.baidu.com/s/1WIxD9l4xKP5UgAiMV7SZjg](https://pan.baidu.com/s/1WIxD9l4xKP5UgAiMV7SZjg),提取码：92cz 

## 3.项目的使用  
可以只运行predict.py文件测试模型的效果，也可以通过运行train.py文件训练自己的模型。  
通过修改config.py文件中的参数来调整模型和语料的处理规则，通过替换data文件夹中的语料文件并且修改相关参数，并针对新的语料文件对data_unit.py进行对应的修改，来训练一个新的聊天机器人。  
## 4. 不足之处  
 - 对原始语料的清洗还不够完善，比如标点符号的处理，少量英文单词的处理。同时原始语料中存在了少量的不雅内容。  
 - 模型过于简单，且由于生成模型的固有特性，导致有的时候回答与问题没有任何关系。
 - 机器人在面对个人信息的询问时，有的时候是自相矛盾的。比如先问机器人"你是男生还是女生？"， 机器人可能会回答"我是男生。"。但接着问 "你是女生吗？"，机器人可能会回答"是的。"。 这个问题可以参考这篇论文进行改善：[Assigning Personality/Identity to a Chatting Machine
for Coherent Conversation Generation
](https://arxiv.org/pdf/1706.02861.pdf)
 
 
 
