#!/usr/bin/env python3

import sys
from ibm import *
from pprint import pprint

def gen_dict(corpus):
    def corpus_to_dict(corpus, acc):
        index = 0
        for sentence in corpus:
            for word in sentence:
                if word not in acc:
                    acc[word] = index
                    index += 1

    lang_dict = {}
    corpus_to_dict(corpus, lang_dict)

    index_to_word = [0] * len(lang_dict)
    for word, index in lang_dict.items():
        index_to_word[index] = word

    return lang_dict, index_to_word


if __name__ == '__main__':
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        sys.exit(-1)

    foreign_corpus_file = sys.argv[1]
    source_corpus_file  = sys.argv[2]

    foreign_corpus = [line.split() for line in open(foreign_corpus_file, 'r')]
    flattened_foreign_corpus = [item for sublist in foreign_corpus for item in sublist]
    foreign_voc_size = len(set(flattened_foreign_corpus))
    # Adding the NULL symbol for the source corpus
    source_corpus  = [['NULL'] + line.split() for line in open(source_corpus_file, 'r')]

    foreign_dict, index_to_foreign = gen_dict(foreign_corpus)
    source_dict, index_to_source = gen_dict(source_corpus)

    # Replace by word indexes
    foreign_corpus = [[foreign_dict[word] for word in sentence] for sentence in foreign_corpus]
    source_corpus = [[source_dict[word] for word in sentence] for sentence in source_corpus]
    
    if len(sys.argv) == 3:
        iterations = 3
        print("IBM model 1")
        model1 = Model(model_setup=Model1Setup(), num_iter=iterations)
        model1.train(foreign_corpus, source_corpus, clear=True)
    else: 
        iterations = 3
        print("IBM model 1 with improvements")
        for n in [1,10,20,50]:
            print("n=" + str(n))
            model1 = Model(model_setup=Model1ImprovedSetup(foreign_voc_size,n), num_iter=iterations)
            model1.train(foreign_corpus, source_corpus, clear=True)
    
    iterations = 3
    print("IBM model 2")
    model2 = Model(model1.t, model1.q, model_setup=Model2Setup(), num_iter=iterations)
    model2.train(foreign_corpus, source_corpus, clear=False)

    with open('debug', 'w') as debug:
        for f, e, i in zip(foreign_corpus, source_corpus, range(len(foreign_corpus))):
            print('# Sentence pair (%s) source length %s target length %s alignment score : %s'
                      % (i + 1, len(e), len(f), model2.translation_score(f, e)), file=debug)
            print(' '.join([index_to_foreign[w_f] for w_f in f]), file=debug)
            alignments = [' '.join([str(index + 1) for index in lst]) for lst in model2.align_viterbi(f, e)]

            print(' '.join([index_to_source[w_e] + ' ({ ' + al  + ' })' for w_e, al in zip(e, alignments)]), file=debug)
