"""
对语料进行观察和处理
"""

from data_unit import DataUnit
import json
import csv
import codecs

if __name__ == '__main__':
    with open('data_config.json', 'r', encoding='utf-8') as fr:
        config = json.load(fr)
    du = DataUnit(**config)
    with open('data.csv', 'w', encoding='utf_8_sig', newline='') as fw:
        writer = csv.writer(fw, dialect='excel')
        writer.writerow(['Question','Answer'])
        for qa in du.data:
            writer.writerow([qa[0], qa[1]])