# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from io import open
import xml.etree.ElementTree
from lxml import etree
from SparseCoding import sparse_coding
import time
import random


INPUT_FILE_PATH = './Pasokh_Dataset/Multi - Dataset'
my_summ_path = './mySumm/tf_multiDoc'
track_list = ['Track1', 'Track2', 'Track3', 'Track4', 'Track5', 'Track6', 'Track7', 'Track8']
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=1, max_df=0.5)
parser = etree.XMLParser(recover=True)

t1 = time.time()
for track in track_list:
    folders = []
    input_path = INPUT_FILE_PATH + '/' + track + '/' + 'Source'
    folders = os.listdir(input_path)
    folders = [e for e in folders if e not in "topics.xml"]
    for folder in folders:
        folder_path = input_path + '/' + folder
        current_out_path = my_summ_path + '/' + track + '/' + folder + '.text'
        xml_files = os.listdir(folder_path)  # 20 xml
        subject_sentences = []
        for xml_file in xml_files:
            current_path = folder_path + '/' + xml_file.title()
            root = etree.fromstring(open(current_path, 'r', encoding="utf-8").read(), parser)
            for child in root:
                sentences = []
                if child.tag == 'TITLE' or child.tag == 'TEXT':
                    if child.text != "":
                        val = child.text.replace('\n', '')
                        sentences = list(re.split('\.', val))
                        subject_sentences.append([e for e in sentences if e not in {u"", u'', u'»', u'«'}])

        subject_sentences = [item for sublist in subject_sentences for item in sublist]
        vectorizer.fit_transform(subject_sentences)
        term_freq = vectorizer.transform(subject_sentences)
        tf_dic = term_freq.todok()
        (row, coll) = (len(subject_sentences), len(vectorizer.get_feature_names()))
        tf_matrix = np.zeros((row, coll))
        for key, value in tf_dic.iteritems():
            (x1, x2) = key
            tf_matrix[x1, x2] = value
        # do sparse coding here

        summ_sen = sparse_coding(subject_sentences, tf_matrix, 'multi')
        out_file = open(current_out_path, 'w', encoding="utf-8")
        for item in summ_sen:
            out_file.write("%s\n" % item)
        out_file.close()


t2 = time.time()
print t2 - t1




