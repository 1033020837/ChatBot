"""
    训练模型
    运行此文件开始训练
    默认训练轮数为200轮
    我在1080的GPU上大概训练了15个小时
    最终得到的loss值大概为0.3
"""

import data_unit
import os
import tensorflow as tf
from seq2seq import Seq2Seq
from tqdm import tqdm
import numpy as np
from config import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config, \
    n_epoch, batch_size, keep_prob

# 是否在原有模型的基础上继续训练
continue_train = False

def train():
    """
    训练模型
    :return:
    """
    du = data_unit.DataUnit(**data_config)
    save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
    steps = int(len(du) / batch_size) + 1

    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 定义模型
            model = Seq2Seq(batch_size = batch_size,
                            encoder_vocab_size = du.vocab_size,
                            decoder_vocab_size = du.vocab_size,
                            mode = 'train',
                            **model_config)
            init = tf.global_variables_initializer()
            sess.run(init)
            if continue_train:
                model.load(sess, save_path)
            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = du.next_batch(batch_size)
                    max_len = np.max(yl)
                    y = y[ : , 0:max_len]
                    cost, lr = model.train(sess, x, xl, y, yl, keep_prob)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(epoch, np.mean(costs), lr))
                model.save(sess, save_path=save_path)

if __name__ == '__main__':
    train()