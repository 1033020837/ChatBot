"""
    对训练好的模型进行测试
    运行此文件开始与机器人聊天
    注意只能进行中文聊天
"""

import data_unit
import os
import tensorflow as tf
from seq2seq import Seq2Seq
import numpy as np
from config import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config

def predict():
    """
    针对用户输入的聊天内容给出回复
    :return:
    """
    du = data_unit.DataUnit(**data_config)
    save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
    batch_size = 1
    tf.reset_default_graph()
    model = Seq2Seq(batch_size=batch_size,
                    encoder_vocab_size=du.vocab_size,
                    decoder_vocab_size=du.vocab_size,
                    mode='decode',
                    **model_config)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.load(sess, save_path)
        while True:
            q = input('请输入聊天内容：')
            if q is None or q.strip() == '':
                print('-----------------------------')
                continue
            if q == r'\b':
                print('再见！')
                exit()
            q = q.strip()
            indexs = du.transform_sentence(q)
            x = np.asarray(indexs).reshape((1,-1))
            xl = np.asarray(len(indexs)).reshape((1,))
            pred = model.predict(
                sess, np.array(x),
                np.array(xl)
            )
            print('Q:   ', du.transform_indexs(x[0]))
            print('A:   ', du.transform_indexs(pred[0]))
            print('-----------------------------')

if __name__ == '__main__':
    predict()