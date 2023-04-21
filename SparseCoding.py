
import numpy as np
from io import open
import operator
import random


def sparse_coding(candidate_set, sentence_matrix, status):
    penalty_parameter = 0.005
    stopping_iteration = 500
    stopping_value = 0.0001
    learning_rate = 1
    if status == 'single':
        proposed_num = 7
    elif status == 'multi':
        proposed_num = 30
    num_sum_sentences = min(len(candidate_set), proposed_num)  # int(0.3 * len(candidate_set)
    saliency_vector_ite = np.zeros(len(candidate_set))
    for m in range(len(candidate_set)):
        saliency_vector_ite[m] = 0  # round(random.random()/25, 2)
    saliency_vector_ite_1 = np.zeros((len(candidate_set)))
    saliency_vector_ite_1 = saliency_vector_ite.copy()
    param = float(1) / float(len(candidate_set))

    ite = 0
    while ite < stopping_iteration:
        x_bar = np.zeros(len(sentence_matrix[0]))
        for i in range(len(candidate_set)):  # for each sentence
            x_bar += saliency_vector_ite[i] * sentence_matrix[i]
        partial_derivatives = np.zeros(len(candidate_set))
        for i in range(len(candidate_set)):
            for j in range(len(candidate_set)):
                partial_derivatives[i] += sum(
                    [a * b for a, b in zip((sentence_matrix[j] - x_bar), (sentence_matrix[i]))])
            partial_derivatives[i] *= param

        index, most_derivation = max(enumerate(abs(i) for i in partial_derivatives), key=operator.itemgetter(1))

        a_param = saliency_vector_ite[index] - (learning_rate * most_derivation)
        sign = np.sign(a_param)
        if a_param == 0.0 or a_param == -0.0:
            sign = 1.0
        saliency_vector_ite_1[index] = sign * max(a_param - penalty_parameter, 0)

        index_ite = np.argpartition(saliency_vector_ite, -num_sum_sentences)[-num_sum_sentences:]
        index_ite_1 = np.argpartition(saliency_vector_ite_1, -num_sum_sentences)[-num_sum_sentences:]

        approx_sen_ite = np.zeros((1, len(sentence_matrix[0])))
        loss_ite = 0
        for j in index_ite:
            approx_sen_ite += saliency_vector_ite[j] * sentence_matrix[j, :]
        for i in range(len(candidate_set)):
            loss_ite += np.linalg.norm((sentence_matrix[i, :] - approx_sen_ite), ord=2)
        loss_ite = 0.5 * param * loss_ite + (penalty_parameter * np.linalg.norm(saliency_vector_ite, ord=1))

        approx_sen_ite_1 = np.zeros((1, len(sentence_matrix[0])))
        loss_ite_1 = 0
        for j in index_ite_1:
            approx_sen_ite_1 += saliency_vector_ite_1[j] * sentence_matrix[j, :]
        for i in range(len(candidate_set)):
            loss_ite_1 += np.linalg.norm((sentence_matrix[i, :] - approx_sen_ite_1), ord=2)
        loss_ite_1 = 0.5 * param * loss_ite_1 + (penalty_parameter * np.linalg.norm(saliency_vector_ite_1, ord=1))

        stopp2 = -1.0 * stopping_value
        if loss_ite_1 - loss_ite < stopping_value:
            break
        ite += 1
        saliency_vector_ite = saliency_vector_ite_1.copy()

    # finish while
    l = np.argpartition(saliency_vector_ite, -num_sum_sentences)[-num_sum_sentences:]
    best_sentences = []
    for k in l:
        best_sentences.append(candidate_set[k])
    print 'ite = ' + str(ite)
    return best_sentences
