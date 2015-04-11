#!/usr/bin/env python3

import sys
from ibm import *

def corpus_to_dict(corpus, acc):
    index = 0
    for sentence in corpus:
        for word in sentence:
            if word not in acc:
                acc[word] = index
                index += 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(-1)

    foreign_corpus_file = sys.argv[1]
    source_corpus_file  = sys.argv[2]

    foreign_corpus = [line.split() for line in open(foreign_corpus_file, 'r')]
    # Adding the NULL symbol for the source corpus
    source_corpus  = [[''] + line.split() for line in open(source_corpus_file, 'r')]

    foreign_dict = {}
    source_dict = {}
    corpus_to_dict(foreign_corpus, foreign_dict)
    corpus_to_dict(source_corpus, source_dict)

    foreign_corpus = [[foreign_dict[word] for word in sentence] for sentence in foreign_corpus]
    source_corpus = [[source_dict[word] for word in sentence] for sentence in source_corpus]

    index_to_foreign = [0] * len(foreign_dict)
    index_to_source = [0] * len(source_dict)

    for word, i in zip(foreign_dict, range(len(foreign_dict))):
        index_to_foreign[i] = word

    for word, j in zip(source_dict, range(len(source_dict))):
        index_to_source[j] = word

    iterations = 3
    print "IBM model 1"
    model1 = Model(model_setup=Model1Setup(), num_iter=iterations)
    model1.train(foreign_corpus, source_corpus, clear=True)
    
    iterations = 3
    print "IBM model 1 with improvements"
    model1 = Model(model_setup=Model1ImprovedSetup(), num_iter=iterations)
    model1.train(foreign_corpus, source_corpus, clear=True)
    
    iterations = 3
    print "IBM model 2"
    model2 = Model(model1.t, model1.q, model_setup=Model2Setup(), num_iter=iterations)
    model2.train(foreign_corpus, source_corpus, clear=False)
