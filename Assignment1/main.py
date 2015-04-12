#!/usr/bin/env python3

import argparse
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

def choices_descriptions():
   return """
--ibm key supports the following: 
    IBM-M1           - IBM-Model 1, Initialized randomly
    IBM-M2-Rand      - IBM-Model 2, Initialized randomly
    IBM-M2-1         - IBM-Model 2, Initialized by parameters from IBM-M1
    IBM-M2-AddN    - IBM-M2-1 with add-n-smoothing
"""


if __name__ == '__main__':
    DESCRIPTION = """
IBM-M1, IBM-M2 Impelementations.
Copyright (c) Minh Ngo, Peter Dekker
"""

    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=choices_descriptions()) 

    parser.add_argument('--foreign', help='Foreign corpus file', required=True) 
    parser.add_argument('--source', help='Source corpus file', required=True) 
    parser.add_argument('--debug', help='Debug file')
    parser.add_argument('--iter-1', dest='iter1', default=3,
                        help='Number of iterations for the first stage', type=int)
    parser.add_argument('--iter-2', dest='iter2', default=3,
                        help='Number of iterations for the second stage', type=int)

    ibm_mode = ['IBM-M1', 'IBM-M2-Rand', 'IBM-M2-1', 'IBM-M2-AddN']
    parser.add_argument('--ibm', choices=ibm_mode, default='IBM-M1', help='IBM Model mode')
    
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    foreign_corpus = [line.split() for line in open(args.foreign, 'r')]

    flattened_foreign_corpus = [item for sublist in foreign_corpus for item in sublist]
    foreign_voc_size = len(set(flattened_foreign_corpus))

    # Adding the NULL symbol for the source corpus
    source_corpus  = [['NULL'] + line.split() for line in open(args.source, 'r')]

    foreign_dict, index_to_foreign = gen_dict(foreign_corpus)
    source_dict, index_to_source = gen_dict(source_corpus)

    # Replace by word indexes
    foreign_corpus = [[foreign_dict[word] for word in sentence] for sentence in foreign_corpus]
    source_corpus = [[source_dict[word] for word in sentence] for sentence in source_corpus]
    
    iterations = args.iter1
    model = None
    if args.ibm == 'IBM-M1' or args.ibm == 'IBM-M2-1':
        print("IBM model 1")
        model1 = Model(model_setup=Model1Setup(), num_iter=iterations)
        model1.train(foreign_corpus, source_corpus, clear=True)
        model = model1
    elif args.ibm == 'IBM-M2-AddN': 
        print("IBM model 1 with improvements")
        for n in [1,10,20,50]:
            print("n=" + str(n))
            model1 = Model(model_setup=Model1ImprovedSetup(foreign_voc_size,n), num_iter=iterations)
            model1.train(foreign_corpus, source_corpus, clear=True)
            model = model1


    iterations = args.iter2
    if args.ibm == 'IBM-M2-Rand':
        print("IBM model 2")
        model2 = Model(model_setup=Model2Setup(), num_iter=iterations)
        model2.train(foreign_corpus, source_corpus, clear=True)
        model = model2
    elif args.ibm == 'IBM-M2-1' or args.ibm == 'IBM-M2-AddN':
        print("IBM model 2")
        model2 = Model(model1.t, model1.q, model_setup=Model2Setup(), num_iter=iterations)
        model2.train(foreign_corpus, source_corpus, clear=False)
        model = model2

    if args.debug != None:
        with open(args.debug, 'w') as debug:
            for f, e, i in zip(foreign_corpus, source_corpus, range(len(foreign_corpus))):
                print('# Sentence pair (%s) source length %s target length %s alignment score : %s'
                          % (i + 1, len(e), len(f), model.translation_score(f, e)), file=debug)
                print(' '.join([index_to_foreign[w_f] for w_f in f]), file=debug)
                alignments = [' '.join([str(index + 1) for index in lst]) for lst in model.align_viterbi(f, e)]

                print(' '.join([index_to_source[w_e] + ' ({ ' + al  + ' })' for w_e, al in zip(e, alignments)]), file=debug)
