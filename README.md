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
 - model 该文件夹用于保存训练好的模型。  
 
 
 
