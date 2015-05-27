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
    'NNS' : 'N',
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

core_to_generic_pos = {}

for key, value in generic_to_core_pos.items():
    if value not in core_to_generic_pos:
        core_to_generic_pos[value] = {key}
    else:
        core_to_generic_pos[value].add(key)


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


def wordtag_score(wordtag_1to1_prob, wordtag_1toN_prob):
    """
        Linear combines two 1to1 and 1toN estimators.
    """
    pos_score = {}

    uniq_keys = set(wordtag_1to1_prob.keys()).union(wordtag_1toN_prob.keys())
    for key in uniq_keys:
        if key in wordtag_1to1_prob:
            score = wordtag_1to1_prob[key]
            if key in wordtag_1toN_prob:
                score += wordtag_1toN_prob[key]
                score /= 2.0
            pos_score[key] = score
        else:
            pos_score[key] = wordtag_1toN_prob[key]
    return pos_score


def corpus_stat(corpus_path, tagger):
    """
        Calculates components of the noisy channel equation
    """
    wordtag_1to1_prob = {}
    wordtag_1toN_prob = {}
    word_count_1to1 = {}
    word_count_1toN = {}
    word_prob= {}
    pos_prob = {}

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
                increase(word_prob, target_word)
                increase(pos_prob, pos_tag)
                if one_to_n_marker[source_ind]:
                    increase(word_count_1toN, target_word)
                    increase(wordtag_1toN_prob, key)
                else:
                    increase(word_count_1to1, target_word)
                    increase(wordtag_1to1_prob, key)

    # Normalizing counters
    for key in wordtag_1to1_prob:
        wordtag_1to1_prob[key] /= word_count_1to1[key[0]]

    for key in wordtag_1toN_prob:
        wordtag_1toN_prob[key] /= word_count_1toN[key[0]]

    word_norm_coef = sum(word_prob.values())
    word_prob = {word : count / word_norm_coef for word, count in word_prob.items()}

    pos_norm_coef = sum(pos_prob.values())
    pos_prob = {pos_tag : count / pos_norm_coef for pos_tag, count in pos_prob.items()}

    return wordtag_score(wordtag_1to1_prob, wordtag_1toN_prob), word_prob, pos_prob


def pos_score(corpus_path, tagger):
    """
        Magic with smoothing...
    """
    wordtag_score, word_prob, pos_prob = corpus_stat(corpus_path, tagger)

    word_to_tags = {}
    for (word, tag), score in wordtag_score.items():
        if word in word_to_tags:
            word_to_tags[word].append((tag, score))
        else:
            word_to_tags[word] = [(tag, score)]

    word_to_core_tags = {}
    for word, tag_scores in word_to_tags.items():
        core_tag_score = {}
        for tag, score in tag_scores:
            core_tag = generic_to_core_pos[tag]
            if core_tag not in core_tag_score:
                core_tag_score[core_tag] = score
            else:
                core_tag_score[core_tag] += score
        word_to_core_tags[word] = list(core_tag_score.items())

    wordtag_score = {}
    for word, core_tags in word_to_core_tags.items():
        toptwo_cores = heapq.nlargest(2, core_tags, lambda x: x[1])
        norm = reduce(lambda x, y: x + y[1] , toptwo_cores, 0)
        for tag, _ in toptwo_cores:
            generic_tags = [(generic_tag, score) for generic_tag, score in word_to_tags[word]
                                                 if generic_tag in core_to_generic_pos[tag]]
            toptwo_generic = heapq.nlargest(2, generic_tags, lambda x: x[1])
            for generic_tag, score in toptwo_generic:
                wordtag_score[(word, generic_tag)] = score / norm

    return wordtag_score, pos_prob, word_prob


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
        print('Score:')
        pprint.pprint(score)
        # TODO Combine multiple tagged corpora

        # TODO: Evaluate the target tags using annotated corpus.

if __name__ == '__main__':
    main()
