# -*- coding: utf-8 -*-

from enum import Enum


class Level(Enum):
    Rouge_1 = 1
    Rouge_2 = 2


def collect_pairs(lines):
    token = []
    for line in lines:
        words = line.split()
        for i in range(len(words) - 1):
            token.append(words[i] + " " + words[i + 1])

    return set(token)


def Rouge(candidate_summary, reference_summary, level, mode):  # mode = precision or recall
    coOccurrings = []
    if level == Level.Rouge_2:
        bigrams = collect_pairs(reference_summary)
        for summary in candidate_summary:
            for bigram in bigrams:
                if bigram in summary:
                    coOccurrings.append(bigram)

        if mode == "precision":
            try:
                return ((float)(len(set(coOccurrings)))) / (len(collect_pairs(candidate_summary)))
            except Exception:
                print("Exception : ", candidate_summary)
                return 0
        elif mode == "recall":
            return ((float)(len(set(coOccurrings)))) / (len(bigrams))
        else:
            return -1

    elif level == Level.Rouge_1:
        splited = []
        for summary in reference_summary:
            splited += summary.split()
        unigrams = set(splited)
        for summary in candidate_summary:
            for unigram in unigrams:
                if unigram in summary.split():
                    coOccurrings.append(unigram)

        if mode == "precision":
            tmp = []
            for summary in candidate_summary:
                tmp += summary.split()
            return ((float)(len(set(coOccurrings)))) / len(set(tmp))
        elif mode == "recall":
            return ((float)(len(set(coOccurrings)))) / len(unigrams)
        else:
            return -1

    else:
        return -1


def rouge_Fscore(candidate_summary, reference_summary, n):
    Precision = Rouge(candidate_summary, reference_summary, n, "precision")
    Recall = Rouge(candidate_summary, reference_summary, n, "recall")

    # print Precision.type()
    if not (Precision + Recall == 0.0):
    # if Precision != 0 and Recall != 0:
        return 2 * (Precision * Recall) / (Precision + Recall)
    else:
        print("No CoOccurrings")
        return 0


def rouge_precision(candidate_summary, reference_summary, n):
    precision = Rouge(candidate_summary, reference_summary, n, "precision")
    return precision


def rouge_recall(candidate_summary, reference_summary, n):
    recall = Rouge(candidate_summary, reference_summary, n, "recall")
    return recall

