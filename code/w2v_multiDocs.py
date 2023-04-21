# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from io import open
import xml.etree.ElementTree
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from lxml import etree
from SparseCoding import sparse_coding
import random


def load_model(model_path):
    wiki_model = KeyedVectors.load_word2vec_format(model_path)
    return wiki_model


INPUT_FILE_PATH = './Pasokh_Dataset/Multi - Dataset'
my_summ_path = './mySumm/w2vec_multiDoc'
track_list = ['Track1', 'Track2', 'Track3', 'Track4', 'Track5', 'Track6', 'Track7', 'Track8']
parser = etree.XMLParser(recover=True)
output_path = './hamshahri.fa.text.vector10.txt'
current_model = load_model(output_path)
wordVectors = current_model.wv  # KeyedVectors Instance gets stored


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

        w2v_matrix = np.zeros((len(subject_sentences), 300))  # len(subject_sentences), 200
        sen_num = 0
        regex = re.compile('[,\.!?:،]')
        for sen in subject_sentences:
            sentence_vector = np.zeros((1, 300))
            k = 0
            for word in sen.split():
                word = regex.sub('', word)
                word2 = filter(lambda x: x in current_model.vocab, [word])
                if len(word2) > 0:
                    k += 1
                    sentence_vector += wordVectors.word_vec(word)
            sentence_vector[:] = [(x / k) for x in sentence_vector]
            w2v_matrix[sen_num, :] = sentence_vector
            sen_num += 1
        # do sparse coding here
        summ_sen = sparse_coding(subject_sentences, w2v_matrix, 'multi')
        out_file = open(current_out_path, 'w', encoding="utf-8")
        for item in summ_sen:
            out_file.write("%s\n" % item)
        out_file.close()
