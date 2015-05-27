#!/usr/bin/env python3

import locale
import sys
import subprocess
import os
import heapq
import pprint

from functools import reduce

from nltk.tag.stanford import POSTagger

corpus_paths = ['../data/en-cs-combined10000.txt']
encoding = locale.getdefaultlocale()[1]

generic_to_core_pos = {
    'NN' : 'N',
    'NNP' : 'N',
    'VB' : 'V',
    'VBP' : 'V',
    'VBG' : 'V',
    'VBN' : 'V',
    'VBD' : 'V',
    'DT' : 'D',
    'WDT' : 'D',
    'CC' : 'C',
    'CD' : 'NUM',
    'RB' : 'R',
    'WRB' : 'R',
    'JJ' : 'J',
    'PRP' : 'P',
    'IN' : 'I'
    }


def increase(collection, key):
    """
        Increases the key counter by 1 or set it to 1 if it doesn't exist.
    """
    if key in collection:
        collection[key] += 1
    else:
        collection[key] = 1


chunk_size = 1000
def parse_corpus(corpus_file, tagger):
    """
        Parses source and target sentences into lexemes, determines POS of the source
        sentence by chunk of chunk_size sentences per time and yields the triple as a
        tuple.
    """
    source_queue = []
    target_queue = []
    iteration = 0
    for corpus_line in corpus_file:
        source_line, target_line = tuple(corpus_line.split(' ||| '))

        # POS tag this source line
        source_words = source_line.split()
        target_words = target_line.split()

        source_queue.append(source_words)
        target_queue.append(target_words)

        if iteration == chunk_size - 1:
            source_tags = tagger.tag_sents(source_queue)
            for source, target, tags in zip(source_queue, target_queue, source_tags):
                yield source, target, tags
            source_queue = []
            target_queue = []

        iteration = (iteration + 1) % chunk_size

    if source_queue:
        source_tags = tagger.tag_sents(source_queue)
        for source, target, tags in zip(source_queue, target_queue, source_tags):
            yield source, target, tags


def mt_alignment(corpus_path):
    """
        Yields a list of alignments for each sentence.
    """
    command = './fast_align/fast_align'
    output = subprocess.check_output([command, '-i', corpus_path])
    alignment_list = output.decode(encoding).split('\n')[:-1]
    del output
    print('Got alignments:', len(alignment_list))

    for alignment in alignment_list:
        yield [tuple(map(int, a.split('-'))) for a in alignment.split()]


def create_stanford_postagger():
    return POSTagger('stanford-postagger-2015-04-20/models/english-bidirectional-distsim.tagger',
                     'stanford-postagger-2015-04-20/stanford-postagger.jar')


def pos_score(corpus_path, tagger):
    wordtag_1to1_prob = {}
    wordtag_1toN_prob = {}
    word_count_1to1 = {}
    word_count_1toN = {}
    word_count = {}
    pos_count = {}

    with open(corpus_path, 'r') as corpus_file:
        for (source_words, target_words, source_tags), alignments in zip(parse_corpus(corpus_file, tagger),
                                                                         mt_alignment(corpus_path)):
            link_count = [0] * len(source_words)
            for source_id, _ in alignments:
                link_count[source_id] += 1
            one_to_n_marker = [count > 1 for count in link_count]

            for (source_ind, _), target_word in zip(alignments, target_words):
                pos_tag = source_tags[source_ind][1]
                key = (target_word, pos_tag)
                increase(word_count, target_word)
                increase(pos_count, pos_tag)
                if one_to_n_marker[source_ind]:
                    increase(word_count_1toN, target_word)
                    increase(wordtag_1toN_prob, key)
                else:
                    increase(word_count_1to1, target_word)
                    increase(wordtag_1to1_prob, key)

    for key in wordtag_1to1_prob:
        wordtag_1to1_prob[key] /= word_count_1to1[key[0]]

    for key in wordtag_1toN_prob:
        wordtag_1toN_prob[key] /= word_count_1toN[key[0]]

    del word_count_1to1
    del word_count_1toN

    word_norm_coef = sum(word_count.values())
    word_count = {word : count / word_norm_coef for word, count in word_count.items()}

    pos_norm_coef = sum(pos_count.values())
    pos_count = {pos_tag : count / pos_norm_coef for pos_tag, count in pos_count.items()}

    wordtag_score = {}

    uniq_keys = set(wordtag_1to1_prob.keys()).union(wordtag_1toN_prob.keys())
    for key in uniq_keys:
        if key in wordtag_1to1_prob:
            score = wordtag_1to1_prob[key]
            if key in wordtag_1toN_prob:
                score += wordtag_1toN_prob[key]
                score /= 2.0
            wordtag_score[key] = score
        else:
            wordtag_score[key] = wordtag_1toN_prob[key]
    print('WordTag score size:', len(wordtag_score))

    del wordtag_1to1_prob
    del wordtag_1toN_prob

    word_to_tags = {}
    for (word, tag), score in wordtag_score.items():
        if word in word_to_tags:
            word_to_tags[word].append((tag, score))
        else:
            word_to_tags[word] = [(tag, score)]

    wordtag_score = {}
    for word, tags in word_to_tags.items():
        toptwo = heapq.nlargest(2, tags, lambda x: x[1])
        norm = reduce(lambda x, y: x + y[1] , toptwo, 0)
        for tag, score in toptwo:
            wordtag_score[(word, tag)] = score / norm

    return wordtag_score


def main():
    # Previous steps done by other programs
    # Load one/multiple parallel corpora
    # Align every parallel corpus

    # Dictionary which contains the tagged target corpora.
    # Every key is a different source corpus.
    print('Creating a POS tagger')
    tagger = create_stanford_postagger()
    print('POS tagger created!')

    for corpus_path in corpus_paths:
        score = pos_score(corpus_path, tagger)
        pprint.pprint(score)
        # TODO Combine multiple tagged corpora

        # TODO: Evaluate the target tags using annotated corpus.

if __name__ == '__main__':
    main()
