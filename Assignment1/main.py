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

    ibm_mode = ['IBM-M1', 'IBM-M2-Rand', 'IBM-M2-1', 'IBM-M1-AddN','IBM-M1-HeavyNull','IBM-M1-HeurInit']
    parser.add_argument('--ibm', choices=ibm_mode, default='IBM-M1', help='IBM Model mode')

    parser.add_argument('--wa', help='Denoted alignment file')
    
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

    gold_alignments = [{'S' : [], 'P' : []} for i in range(len(foreign_corpus))]
    alignment_count = {'S' : 0, 'P' : 0}

    if args.wa != None:
        # Calculate AER
        with open(args.wa, 'r') as wa:
            for line in wa:
                lexemes = line.split()
                sentence_id = int(lexemes[0])
                f_align = int(lexemes[1])
                e_align = int(lexemes[2])
                type_align = lexemes[3]
                gold_alignments[sentence_id - 1][type_align].append((f_align, e_align))
                alignment_count[type_align] += 1

    def stat_calculate(model):
        perplexity = compute_perplexity([model.translation_score_normalized(f, e)
                                         for f, e in zip(foreign_corpus, source_corpus)])
        log_likelihood = compute_log_likelihood([model.translation_prob(f, e)
                                                 for f, e in zip(foreign_corpus, source_corpus)])

        stat = {'A' : 0, 'A & P' : 0, 'A & S': 0}
        for f, e, gold_alignment in zip(foreign_corpus, source_corpus, gold_alignments):
            viterbi_alignment = model.align_viterbi(f, e)
            stat['A'] += len(f)
            for i in range(len(viterbi_alignment)):
                # i -> viterbi_alignment[i]
                word_alignment = (i, viterbi_alignment[i])
                for alignment in gold_alignment['S']:
                    if word_alignment == alignment:
                        stat['A & S'] += 1

                for alignment in gold_alignment['P']:
                    if word_alignment == alignment:
                        stat['A & P'] += 1

        recall = stat['A & S'] / alignment_count['S']
        precision = stat['A & P'] / stat['A']
        aer = 1 - (stat['A & S'] + stat['A & P']) / (stat['A'] + alignment_count['S'])
        print('Perplexity = %s, Log-likelihood = %s, Recall = %s, Precision = %s, AER = %s'
                % (perplexity, log_likelihood, recall, precision, aer))
    
    iterations = args.iter1
    model = None
    if args.ibm == 'IBM-M1' or args.ibm == 'IBM-M2-1':
        print("IBM model 1")
        model1 = Model(model_setup=Model1Setup(), num_iter=iterations)
        model1.train(foreign_corpus, source_corpus, clear=True, callback=stat_calculate)
        model = model1
    elif args.ibm == 'IBM-M1-AddN': 
        print("IBM model 1 with add-n smoothing")
        for n in [1,10,20,50]:
            print("n=" + str(n))
            model1 = Model(model_setup=Model1ImprovedSetup(0,foreign_voc_size,n), num_iter=iterations)
            model1.train(foreign_corpus, source_corpus, clear=True, callback=stat_calculate)
            model = model1
    elif args.ibm == 'IBM-M1-HeavyNull': 
        print("IBM model 1 with more weight on null alignment")
        model1 = Model(model_setup=Model1ImprovedSetup(1), num_iter=iterations)
        model1.train(foreign_corpus, source_corpus, clear=True, callback=stat_calculate)
        model = model1
    elif args.ibm == 'IBM-M1-HeurInit': 
        print("IBM model 1 with heuristic initialization")
        print("n=" + str(n))
        model1 = Model(model_setup=Model1ImprovedSetup(2), num_iter=iterations)
        model1.train(foreign_corpus, source_corpus, clear=True, callback=stat_calculate)
        model = model1


    iterations = args.iter2
    if args.ibm == 'IBM-M2-Rand':
        print("IBM model 2")
        model2 = Model(model_setup=Model2Setup(), num_iter=iterations)
        model2.train(foreign_corpus, source_corpus, clear=True, callback=stat_calculate)
        model = model2
    elif args.ibm == 'IBM-M2-1' or args.ibm == 'IBM-M2-AddN':
        print("IBM model 2")
        model2 = Model(model1.t, model1.q, model_setup=Model2Setup(), num_iter=iterations)
        model2.train(foreign_corpus, source_corpus, clear=False, callback=stat_calculate)
        model = model2

    if args.debug != None:
        with open(args.debug, 'w') as debug:
            for f, e, i in zip(foreign_corpus, source_corpus, range(len(foreign_corpus))):
                print("# Sentence pair (%s) source length %s target length %s alignment score : %s" % (i + 1, len(e), len(f), model.translation_score_normalized(f, e)), file=debug)
                print(' '.join([index_to_foreign[w_f] for w_f in f]), file=debug)

                f_to_e_alignment = model.align_viterbi(f, e)
                e_to_f_alignment = [[] for i in range(len(e))]
                for i in range(len(f_to_e_alignment)):
                    e_to_f_alignment[f_to_e_alignment[i]].append(i)

                alignments = [' '.join([str(index + 1) for index in lst]) for lst in e_to_f_alignment]

                print(' '.join([index_to_source[w_e] + ' ({ ' + al  + ' })' for w_e, al in zip(e, alignments)]), file=debug)
