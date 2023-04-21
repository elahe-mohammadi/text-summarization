# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import logging
import os
import re
import numpy as np
from io import open
import operator
import random
from SparseCoding import sparse_coding


def load_model(model_path):
    wiki_model = KeyedVectors.load_word2vec_format(model_path)
    return wiki_model


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    INPUT_FILE_PATH = './Pasokh_Dataset/Single -Dataset/Source/DUC'
    my_summ_path = './mySumm/w2vec_singleDoc'
    single_docs = os.listdir(INPUT_FILE_PATH)
    output_path = './hamshahri.fa.text.vector10.txt'
    current_model = load_model(output_path)
    wordVectors = current_model.wv  # KeyedVectors Instance gets stored

    for doc in single_docs:
        current_path = INPUT_FILE_PATH + '/' + doc.title()
        current_out_path = my_summ_path + '/' + doc.title()

        the_file = open(current_path, 'r', encoding="utf-8")
        data = the_file.read().replace('\n', '')
        sentences = list(re.split('\.', data))
        sentences = [e for e in sentences if e not in u""]

        w2v_matrix = np.zeros((len(sentences), 300))
        regex = re.compile('[,\.!?:،]')
        sen_num = 0
        for sen in sentences:
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
        summ_sen = sparse_coding(sentences, w2v_matrix, 'single')
        out_file = open(current_out_path, 'w', encoding="utf-8")
        for item in summ_sen:
            out_file.write("%s\n" % item)
        out_file.close()
