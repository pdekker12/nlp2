#!/usr/bin/env python3

import argparse
import sys
from ibm import *
from evaluation import compute_perplexity, compute_log_likelihood
from pprint import pprint

test_length=0

def corpus_to_dict(corpus, acc):
    index = 0
    for sentence in corpus:
        for word in sentence:
            if word not in acc:
                acc[word] = index
                index += 1

def gen_dict(corpus):
    lang_dict = {}
    corpus_to_dict(corpus, lang_dict)

    index_to_word = [0] * len(lang_dict)
    for word, index in lang_dict.items():
        index_to_word[index] = word

    return lang_dict, index_to_word

def export_weights(file_name, model):
    print('Exporting weights...')
    with open(file_name + '.t', 'w') as output:
        for key, weight in model.t.items():
            print('%d %f' % (key, weight), file=output)

    with open(file_name + '.q', 'w') as output:
        for key, weight in model.q.items():
            print('%d %f' % (key, weight), file=output)

def import_weights(file_name):
    print('Importing weights...')
    t = {}
    q = {}
    with open(file_name + '.t', 'r') as input_file:
        for line in input_file:
            lexemes = line.split()
            t[int(lexemes[0])] = float(lexemes[-1])

    with open(file_name + '.q', 'r') as input_file:
        for line in input_file:
            lexemes = line.split()
            q[int(lexemes[0])] = float(lexemes[-1])

    return t, q

def choices_descriptions():
   return """
--ibm key supports the following: 
    IBM-M1           - IBM-Model 1, Initialized randomly
    IBM-M1-AddN      - IBM-M2-1 with add-n-smoothing
    IBM-M2-Rand      - IBM-Model 2, Initialized randomly
    IBM-M2-1         - IBM-Model 2, Initialized by parameters from IBM-M1
    IBM-M2-Uniform   - IBM-Model 2, Initialized uniformly
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

    ibm_mode = ['IBM-M1', 'IBM-M2-Rand', 'IBM-M2-1', 'IBM-M2-Uniform',
                'IBM-M1-AddN','IBM-M1-HeavyNull','IBM-M1-HeurInit','IBM-M1-SmoothHeavyNull','IBM-M1-AllImprove']
    parser.add_argument('--ibm', choices=ibm_mode, default='IBM-M1', help='IBM Model mode')

    parser.add_argument('--wa', help='Denoted alignment file')
    parser.add_argument('--output', help='Output result file')

    parser.add_argument('--export', help='Export weights')
    parser.add_argument('--import', dest='import_file', help='Import weights')
    
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    foreign_corpus = [line.split() for line in open(args.foreign, 'r')]

    flattened_foreign_corpus = [item for sublist in foreign_corpus for item in sublist]
    foreign_voc_size = len(set(flattened_foreign_corpus))

    # Adding the NULL symbol for the source corpus
    source_corpus  = [['NULL'] + line.split() for line in open(args.source, 'r')]
    
    
    # Remove sentence pairs where source or foreign sentence has length > 100
    short_source_corpus = []
    short_foreign_corpus = []
    for f,e in zip(foreign_corpus, source_corpus):
        if len(f) < MAX_SENTENCE_LENGTH and len(e) < MAX_SENTENCE_LENGTH:
            short_source_corpus.append(e)
            short_foreign_corpus.append(f)
        
    foreign_dict, index_to_foreign = gen_dict(short_foreign_corpus)
    source_dict, index_to_source = gen_dict(short_source_corpus)

    # Replace by word indexes
    foreign_corpus = [[foreign_dict[word] for word in sentence] for sentence in short_foreign_corpus]
    source_corpus = [[source_dict[word] for word in sentence] for sentence in short_source_corpus]
    
    # Calculate test_length: highest sentence number in evaluation file
    if args.wa != None:
        with open(args.wa, 'r') as wa:
            for line in wa:
                sid = int(line.split()[0])
                if sid > test_length:
                    test_length = sid


    gold_alignments = [{'S' : [], 'P' : []} for i in range(test_length)]
    alignment_count_s = 0

    imported_t = None
    imported_q = None
    if args.import_file != None:
        imported_t, imported_q = import_weights(args.import_file)

    if args.wa != None:
        # Calculate AER
        with open(args.wa, 'r') as wa:
            for line in wa:
                lexemes = line.split()
                sentence_id = int(lexemes[0])
                e_align = int(lexemes[1])
                f_align = int(lexemes[2])
                type_align = lexemes[3]
                gold_alignments[sentence_id - 1][type_align].append((f_align, e_align))
                if type_align == 'S':
                    alignment_count_s += 1

    def stat_calculate(model):
        score = [model.translation_score_normalized(f, e)
                 for f, e in zip(foreign_corpus, source_corpus)]
        perplexity = compute_perplexity(score)
        log_likelihood = compute_log_likelihood(score)
        print('Perplexity = %s, Log-Likelihood = %s' % (perplexity, log_likelihood))

        if args.wa:
            stat_a = 0
            stat_a_and_p = 0
            stat_a_and_s = 0
            for f, e, gold_alignment in zip(foreign_corpus[:test_length], source_corpus[:test_length], gold_alignments):
                viterbi_alignment = model.align_viterbi(f, e)
                gold_alignment_s = gold_alignment['S']
                gold_alignment_p = gold_alignment['P']
                for i in range(len(f)):
                    # i -> viterbi_alignment[i]
                    if viterbi_alignment[i] == 0:
                        continue
                    stat_a += 1

                    word_alignment = (i + 1, viterbi_alignment[i])

                    for alignment in gold_alignment_s:
                        if word_alignment == alignment:
                            stat_a_and_s += 1
                            stat_a_and_p += 1
                            break

                    for alignment in gold_alignment_p:
                        if word_alignment == alignment:
                            stat_a_and_p += 1
                            break

            recall = stat_a_and_s / alignment_count_s
            precision = stat_a_and_p / stat_a
            aer = 1 - (stat_a_and_s + stat_a_and_p) / (stat_a + alignment_count_s)
            print('Recall = %s, Precision = %s, AER = %s' % (recall, precision, aer))
    
    iterations = args.iter1
    model = None

    def train_model1(model1):
        global model
        if args.import_file:
            model1.t = imported_t
            model1.q = imported_q
        model1.train(foreign_corpus, source_corpus, clear=(args.import_file == None), callback=stat_calculate)
        model = model1

    if args.ibm == 'IBM-M1' or args.ibm == 'IBM-M2-1':
        print('IBM model 1')
        model1 = Model(model_setup=Model1Setup(), num_iter=iterations)
        train_model1(model1)
    elif args.ibm == 'IBM-M1-AddN': 
        print('IBM model 1 with add-n smoothing')
        #for n in [5]:
        #    for v in [0.7,1]:
        #        print('n=' + str(n))
        #        print('v=' + str(v * foreign_voc_size))
        model1 = Model(model_setup=Model1ImprovedSetup(0,voc_size=0.7*foreign_voc_size,add_n=2), num_iter=iterations)
        train_model1(model1)
    elif args.ibm == 'IBM-M1-HeavyNull': 
        print('IBM model 1 with more weight on null alignment')
        #for w in [2,3,5,10]:
        #    print('null_weight=' + str(w))
        #    model1 = Model(model_setup=Model1ImprovedSetup(1,null_weight=w), num_iter=iterations)
        #    train_model1(model1)
        model1 = Model(model_setup=Model1ImprovedSetup(1,null_weight=10), num_iter=iterations)
        train_model1(model1)
    elif args.ibm == 'IBM-M1-HeurInit': 
        print('IBM model 1 with heuristic initialization')
        init_model = InitModel()
        # Get heuristically initialized t from init model
        # v and n are best values of running only AddN extension on small test corpus
        t_heur = init_model.train(foreign_corpus,source_corpus)
        model1 = Model(t=t_heur, model_setup=Model1ImprovedSetup(2), num_iter=iterations)
        train_model1(model1)
    elif args.ibm == 'IBM-M1-SmoothHeavyNull': 
        print('IBM model 1 with smoothing and heavy null')
        init_model = InitModel()
        v = 0.7* foreign_voc_size
        n = 2
        w = 10
        model1 = Model(model_setup=Model1ImprovedSetup(3,voc_size=v,add_n=n,null_weight=w), num_iter=iterations)
        train_model1(model1)
    elif args.ibm == 'IBM-M1-AllImprove': 
        print('IBM model 1 with all Moore improvements')
        init_model = InitModel()
        # Get heuristically initialized t from init model
        t_heur = init_model.train(foreign_corpus,source_corpus)
        v = 0.7* foreign_voc_size
        n = 2
        w=10
        model1 = Model(t=t_heur, model_setup=Model1ImprovedSetup(4,voc_size=v,add_n=n,null_weight=w), num_iter=iterations)
        train_model1(model1)


    print('Model 1 instance:', model)
    iterations = args.iter2
    if args.ibm == 'IBM-M2-Rand' or args.ibm == 'IBM-M2-Uniform':
        print('IBM model 2 with random weights')
        model2 = Model(model_setup=Model2Setup(), num_iter=iterations)

        if args.import_file:
            model2.t = imported_t
            model2.q = imported_q

        model2.train(foreign_corpus, source_corpus, clear=(args.import_file == None),
                     callback=stat_calculate, uniform=(args.ibm == 'IBM-M2-Uniform'))
        model = model2
    elif args.ibm == 'IBM-M2-1':
        print('IBM model 2 initialized by IBM-M1')
        model2 = Model(t=model.t, q=model.q, model_setup=Model2Setup(), num_iter=iterations)
        model2.train(foreign_corpus, source_corpus, clear=False, callback=stat_calculate)
        model = model2

    def parallel_corpus(foreign_corpus, source_corpus):
        for f, e, i in zip(foreign_corpus, source_corpus, range(len(foreign_corpus))):
            yield (f, e, i)

    if args.output != None:
        with open(args.output, 'w') as output:
            for f, e, k in parallel_corpus(foreign_corpus, source_corpus):
                viterbi_alignment = model.align_viterbi(f, e)
                for i, j in zip(range(len(viterbi_alignment)), viterbi_alignment):
                    if j != 0:
                        print('%04d %d %d' % (k + 1, j, i + 1),file=output)

    if args.export != None:
        export_weights(args.export, model)

    if args.debug != None:
        print('Generating the debug file...')
        with open(args.debug, 'w') as debug:
            for f, e, i in parallel_corpus(foreign_corpus, source_corpus):
                print('# Sentence pair (%s) source length %s target length %s alignment score : %s'
                        % (i + 1, len(e), len(f), model.translation_score_normalized(f, e)), file=debug)
                print(' '.join(index_to_foreign[w_f] for w_f in f), file=debug)

                f_to_e_alignment = model.align_viterbi(f, e)
                e_to_f_alignment = [[] for i in range(len(e))]
                for i in range(len(f_to_e_alignment)):
                    e_to_f_alignment[f_to_e_alignment[i]].append(i)

                alignments = [' '.join(str(index + 1) for index in lst) for lst in e_to_f_alignment]

                print(' '.join('%s ({ %s })' % (index_to_source[w_e], al)
                               for w_e, al in zip(e, alignments)), file=debug)
