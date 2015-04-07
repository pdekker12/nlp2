#!/usr/bin/env python3

import sys
from ibm import Model1, Model2

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(-1)

    foreign_corpus_file = sys.argv[1]
    source_corpus_file  = sys.argv[2]

    foreign_corpus = [line.split() for line in open(foreign_corpus_file, 'r')]

    # TODO: Map to the digit dictionary

    # Adding the NULL symbol for the source courpus
    source_corpus  = [[''] + line.split() for line in open(source_corpus_file, 'r')]
                
    iterations = 3
    model1 = Model1(iterations)
    model1.train(foreign_corpus, source_corpus)

    iterations = 1
    model2 = Model2(model1.t, model1.q, iterations)
    model2.train(foreign_corpus, source_corpus)