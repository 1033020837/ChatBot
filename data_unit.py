"""
    处理原始语料数据
    生成批训练数据
"""

from pprint import pprint
import re
import os
import pickle
import json
import collections
import itertools
import random
import numpy as np

class DataUnit(object):

    #特殊标签
    PAD = '<PAD>'
    UNK = '<UNK>'
    START = '<SOS>'
    END = '<EOS>'

    START_INDEX = 0
    END_INDEX =1
    UNK_INDEX = 2
    PAD_INDEX = 3

    def __init__(self, path, processed_path,
                 min_q_len, max_q_len,
                 min_a_len, max_a_len,
                 index2word_path, word2index_path):
        """
        :param path:原始语料库路径
        """
        self.path = path
        self.processed_path = processed_path    #处理过后的语料的存储路径
        self.index2word_path = index2word_path
        self.word2index_path = word2index_path
        self.min_q_len = min_q_len
        self.max_q_len = max_q_len
        self.min_a_len = min_a_len
        self.max_a_len = max_a_len
        self.vocab_size = 0
        self.index2word = {}
        self.word2index = {}
        self.data = self.load_data()
        self._fit_data_()

    def next_batch(self, batch_size):
        data_batch = random.sample(self.data, batch_size)
        batch = []
        for qa in data_batch:
            encoded_q = self.transform_sentence(qa[0])
            encoded_a = self.transform_sentence(qa[1])
            q_len = len(encoded_q)
            encoded_q = encoded_q + [self.func_word2index(self.PAD)] * (self.max_q_len - q_len)
            encoded_a = encoded_a + [self.func_word2index(self.END)]
            a_len = len(encoded_a)
            encoded_a = encoded_a + [self.func_word2index(self.PAD)] * (self.max_a_len + 1 - a_len)
            batch.append((encoded_q, q_len, encoded_a, a_len))
        batch = zip(*batch)
        batch = [np.asarray(x) for x in batch]
        return batch

    def transform_sentence(self, sentence):
        """
        将句子转化为索引
        :param sentence:
        :return:
        """
        res = []
        for word in sentence:
            res.append(self.func_word2index(word))
        return res

    def transform_indexs(self, indexs):
        """
        将索引转化为句子,去除标签
        :param indexs:
        :return:
        """
        res = []
        for index in indexs:
            if index == self.START_INDEX or index == self.PAD_INDEX \
                or index == self.END_INDEX or index == self.UNK_INDEX:
                continue
            res.append(self.func_index2word(index))
        return ''.join(res)

    def _fit_data_(self):
        """
        将语料库与词表对应
        :return:
        """
        if not os.path.exists(self.index2word_path) or not os.path.exists(self.word2index_path):
            vocabularies = [x[0] + x[1] for x in self.data]
            self._fit_word_(itertools.chain(*vocabularies))
            with open(self.index2word_path, 'wb') as fw:
                pickle.dump(self.index2word, fw)
            with open(self.word2index_path, 'wb') as fw:
                pickle.dump(self.word2index, fw)
        else:
            with open(self.index2word_path, 'rb') as fr:
                self.index2word = pickle.load(fr)
            with open(self.word2index_path, 'rb') as fr:
                self.word2index = pickle.load(fr)
        self.vocab_size = len(self.word2index)


    def load_data(self):
        """
        获取处理后的语料
        :return:
        """
        if not os.path.exists(self.processed_path):
            data = self._extract_data()
            with open(self.processed_path, 'wb') as fw:
                pickle.dump(data, fw)
        else:
            with open(self.processed_path, 'rb') as fr:
                data = pickle.load(fr)
        data = [x for x in data if self.min_q_len <= len(x[0]) <= self.max_a_len and self.min_a_len <= len(x[1]) <= self.max_a_len]
        return data

    def func_word2index(self, word):
        return self.word2index.get(word, self.word2index[self.UNK])

    def func_index2word(self, index):
        return self.index2word.get(index, self.UNK)

    def _fit_word_(self, vocabularies):
        """
        获取词语索引的对应
        :param vocabularies:
        :return:
        """
        vocab_counter = collections.Counter(vocabularies)
        index2word = [self.START] + [self.END] + [self.UNK] + [self.PAD] + [x[0] for x in vocab_counter if vocab_counter.get(x[0]) > 4]
        self.word2index = dict([(w, i) for i, w in enumerate(index2word)])
        self.index2word =dict([(i, w) for i, w in enumerate(index2word)])


    def _regular_(self, sen):
        """
        句子规范化
        :param sen:
        :return:
        """
        sen = sen.replace('/', '')
        sen = re.sub(r'…{1,100}', '···', sen)
        sen = re.sub(r'\.{3,100}', '···', sen)
        sen = re.sub(r'···{2,100}', '···', sen)
        sen = re.sub(r',{1,100}', '，', sen)
        sen = re.sub(r'，{1,100}', '，', sen)
        sen = re.sub(r'\.{1,100}', '。', sen)
        sen = re.sub(r'。{1,100}', '。', sen)
        sen = re.sub(r'\?{1,100}', '？', sen)
        sen = re.sub(r'？{1,100}', '？', sen)
        sen = re.sub(r'!{1,100}', '！', sen)
        sen = re.sub(r'！{1,100}', '！', sen)
        sen = re.sub(r'~{1,100}', '～', sen)
        sen = re.sub(r'～{1,100}', '～', sen)
        sen = re.sub(r'０', '0', sen)
        sen = re.sub(r'３', '3', sen)
        sen = re.sub(r'\s{1,100}', '，', sen)
        sen = re.sub(r'[“”]{1,100}', '"', sen)  #中文引号不好处理
        sen = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)
        sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)

        return sen

    def _good_line_(self, line):
        """
        判断一句话是否是好的语料,暂定中文占80%以上为好的
        :param line:
        :return:
        """
        if len(line) == 0:
            return False
        ch_count = 0
        for c in line:
            # 中文字符范围
            if '\u4e00' <= c <= '\u9fff':
                ch_count += 1
        if ch_count / float(len(line)) >= 0.8 and len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) < 3 \
                and len(re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3:
            return True
        return False

    def _extract_data(self):
        """
        从conv文件中读取问答对
        :return:
        """
        res = []
        q = None
        with open(self.path, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line.startswith('M '):
                    if q is None:
                        q = self._regular_(line[2:-1])
                    else:
                        #判断是否是好的问答对
                        a = self._regular_(line[2:-1])
                        if self._good_line_(q) and self._good_line_(a):
                            res.append((q, a))
                        q = None
        return res

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    with open('data_config.json', 'r', encoding='utf-8') as fr:
        config = json.load(fr)
    du = DataUnit(**config)
    print(len(du.data))

