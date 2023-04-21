#from rouge import Rouge
import os
from io import open
import re
from evaluation import rouge_recall, rouge_Fscore, rouge_precision
from evaluation import Level


INPUT_FILE_PATH = './mySumm/w2vec_multiDoc\w2vMultiDoc'
mySum_docs = os.listdir(INPUT_FILE_PATH)

in_path = './realySumm_multi'
single_extractive = os.listdir(in_path)

rouge1_list = []  # for each doc
rouge2_list = []  # for each doc
# r = Rouge()
for doc in mySum_docs:
    rouge1_subList = []  # a list of list of precision, recall, f_measure
    rouge2_subList = []
    current_path = INPUT_FILE_PATH + '/' + doc.title()
    the_file = open(current_path, 'r', encoding="utf-8")
    data = the_file.read().replace('\n', '.')
    candid_sens = list(re.split('\.', data))
    candid_sens = [e for e in candid_sens if e not in u""]

    extra_of_doc = []
    for extra in single_extractive:
        if doc.title().replace('.Text', '') in extra.title():
            extra_of_doc.append(extra)

    for ex in extra_of_doc:
        ex_sentence = ''
        ex_file = open(in_path + '/' + ex.title(), 'r', encoding="utf-8")
        ex_sentence = ex_file.read().replace('\n', '')
        ref_sentences = list(re.split('\.', ex_sentence))
        ref_sentences = [e for e in ref_sentences if e not in u""]
        # print r.get_scores(str(data), str(ex_sentence), avg= True)
        #
        # hyps = [line[:-1] for line in sentences]
        #
        # refs = [line[:-1] for line in my_sen]

        # r1_dic = r.get_scores(hyps, refs, avg=True).get('rouge-1')
        # test = rouge_precision(candidate_summary=candid_sens, reference_summary=ref_sentences, n=Level.Rouge_1)

        l1 = [
            float(rouge_precision(candidate_summary=candid_sens, reference_summary=ref_sentences, n=Level.Rouge_1)),
             float(rouge_recall(candidate_summary=candid_sens, reference_summary=ref_sentences, n=Level.Rouge_1)),
             float(rouge_Fscore(candidate_summary=candid_sens, reference_summary=ref_sentences, n=Level.Rouge_1))]
        rouge1_subList.append(l1)
        # r2_dic = r.get_scores(sentences, my_sen, avg=True).get('rouge-2')
        l2 = [rouge_precision(candidate_summary=candid_sens, reference_summary=ref_sentences, n=Level.Rouge_2),
                  rouge_recall(candidate_summary=candid_sens, reference_summary=ref_sentences, n=Level.Rouge_2),
                  rouge_Fscore(candidate_summary=candid_sens, reference_summary=ref_sentences, n=Level.Rouge_2)]
        rouge2_subList.append(l2)

    avg1 = [0, 0, 0]
    avg2 = [0, 0, 0]
    size = len(rouge1_subList)
    for i in range(size):
        avg1 = [sum(x) for x in zip(avg1, rouge1_subList[i])]
        avg2 = [sum(x) for x in zip(avg2, rouge2_subList[i])]
    avg1[:] = [x / size for x in avg1]
    avg2[:] = [x / size for x in avg2]

    rouge1_list.append(avg1)
    rouge2_list.append(avg2)

rose1 = [0, 0, 0]
rose2 = [0, 0, 0]
size = len(rouge1_list)
for i in range(size):
    rose1 = [sum(x) for x in zip(rose1, rouge1_list[i])]
    rose2 = [sum(x) for x in zip(rose2, rouge2_list[i])]
rose1 = [x / size for x in rose1]
rose2 = [x / size for x in rose2]
print 'Rouge1 = ',  rose1
print 'Rouge2 = ',  rose2





