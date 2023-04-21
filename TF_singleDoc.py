# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from io import open
from SparseCoding import sparse_coding
import operator
import random
from pyrouge import Rouge155


INPUT_FILE_PATH = './Pasokh_Dataset/Single -Dataset/Source/DUC'
my_summ_path = './mySumm/tf_singleDoc'

single_docs = os.listdir(INPUT_FILE_PATH)
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=1, max_df=0.5)

in_path = './Pasokh_Dataset/Single -Dataset/Summ/Extractive'
single_extractive = os.listdir(in_path)

for doc in single_docs:
    current_path = INPUT_FILE_PATH + '/' + doc.title()
    current_output_path = my_summ_path + '/' + doc.title()
    the_file = open(current_path, 'r', encoding="utf-8")
    # sentences = LineSentence(current_path)
    data = the_file.read().replace('\n', '')
    sentences = list(re.split('\.', data))
    sentences = [e for e in sentences if e not in u""]
    middle1 = []
    for sen in sentences:
        middle1.append(re.split('!', sen))
    sentences = [item for sublist in middle1 for item in sublist]
    middle1 = []
    for sen in sentences:
        middle1.append(re.split('؟', sen))
    sentences = [item for sublist in middle1 for item in sublist]

    vectorizer.fit_transform(sentences)
    term_freq = vectorizer.transform(sentences)
    tf_dic = term_freq.todok()
    (row, coll) = (len(sentences), len(vectorizer.get_feature_names()))
    tf_matrix = np.zeros((row, coll))
    for key, value in tf_dic.iteritems():
        (x1, x2) = key
        tf_matrix[x1, x2] = value
    # do sparse coding here
    summ_sen = sparse_coding(sentences, tf_matrix, 'single')
    out_file = open(current_output_path, 'w', encoding="utf-8")
    for item in summ_sen:
        out_file.write("%s\n" % item)
    out_file.close()


