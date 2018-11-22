"""
训练模型
"""
import json
import data_unit
import os
import tensorflow as tf
from seq2seq import Seq2Seq
from tqdm import tqdm
import numpy as np

BASE_MODEL_DIR = 'model'

def train():
    continue_train = True
    with open('data_config.json', 'r', encoding='utf-8') as fr:
        data_config = json.load(fr)
    with open('model_config.json', 'r', encoding='utf-8') as fr:
        model_config = json.load(fr)
    du = data_unit.DataUnit(**data_config)
    save_path = os.path.join(BASE_MODEL_DIR, 'chatbot_model2.ckpt')
    # 训练
    n_epoch = 500
    batch_size = 128
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
                    cost, lr = model.train(sess, x, xl, y, yl)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(epoch, np.mean(costs), lr))
                model.save(sess, save_path=save_path)

if __name__ == '__main__':
    train()