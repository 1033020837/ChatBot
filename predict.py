import json
import data_unit
import os
import tensorflow as tf
from seq2seq import Seq2Seq
import numpy as np

BASE_MODEL_DIR = 'model'

def predict():
    with open('data_config.json', 'r', encoding='utf-8') as fr:
        data_config = json.load(fr)
    with open('model_config.json', 'r', encoding='utf-8') as fr:
        model_config = json.load(fr)
    du = data_unit.DataUnit(**data_config)
    save_path = os.path.join(BASE_MODEL_DIR, 'chatbot_model2.ckpt')
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
        '''
        for _ in range(10):
            x, xl, y, yl = du.next_batch(batch_size)
            pred = model.predict(
                sess, np.array(x),
                np.array(xl)
            )
            print('Question:   ',du.transform_indexs(x[0]))
            print('Real Answer:   ',du.transform_indexs(y[0]))
            print('Predict Answer:   ',du.transform_indexs(pred[0]))
            print('-----------------------------')
        '''
        while True:
            q = input('请输入：')
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