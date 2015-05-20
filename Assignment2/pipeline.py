#!/usr/bin/env python3

import locale
import sys
import subprocess
import os
import heapq

from nltk.tag.stanford import POSTagger

corpus_paths = ['../data/en-cs-combined10000.txt']
encoding = locale.getdefaultlocale()[1]

def increase(collection, key):
    if key in collection:
        collection[key] += 1
    else:
        collection[key] = 1

def main():
    # Previous steps done by other programs
    # Load one/multiple parallel corpora
    # Align every parallel corpus

    # Dictionary which contains the tagged target corpora.
    # Every key is a different source corpus.
    print('Creating a POS tagger')
    tagger = POSTagger('stanford-postagger-2015-04-20/models/english-bidirectional-distsim.tagger',
                       'stanford-postagger-2015-04-20/stanford-postagger.jar')
    print('POS tagger created!')

    for corpus_path in corpus_paths:
        corpus_path_id = corpus_paths.index(corpus_path)

        wordtag_1to1_prob = {}
        wordtag_1toN_prob = {}
        word_count_1to1 = {}
        word_count_1toN = {}

        # Perform alignment
        command = './fast_align/fast_align'
        output = subprocess.check_output([command, '-i', corpus_path])
        alignment_list = output.decode(encoding).split('\n')[:-1]
        del output
        print('Got alignments:', len(alignment_list))

        print('Openning the corpus', corpus_path)
        # Open corpus
        with open(corpus_path, 'r') as corpus_file:
            for corpus_line, alignment in zip(corpus_file, alignment_list):
                print(corpus_line)
                source_line, target_line = tuple(corpus_line.split(' ||| '))

                # POS tag this source line
                source_words = source_line.split()
                target_words = target_line.split()
                source_tags = tagger.tag(source_words)[0]

                # Map source POS tags to target POS tags using alignment.
                # TODO: Use smoothing.
                # Get alignments for this line
                alignments = [tuple(map(int, a.split('-'))) for a in alignment.split()]

                link_count = [0] * len(source_words)
                for source_id, _ in alignments:
                    link_count[source_id] += 1
                one_to_n_marker = [count > 1 for count in link_count]

                for (source_ind, _), target_word in zip(alignments, target_words):
                    pos_tag = source_tags[source_ind][1]
                    key = (target_word, pos_tag)
                    if one_to_n_marker[source_ind]:
                        increase(word_count_1toN, target_word)
                        increase(wordtag_1toN_prob, key)
                    else:
                        increase(word_count_1to1, target_word)
                        increase(wordtag_1to1_prob, key)

        del alignment_list
        for key in wordtag_1to1_prob:
            wordtag_1to1_prob[key] /= word_count_1to1[key[0]]
        print('wordtag_1to1_prob size:', len(wordtag_1to1_prob))

        for key in wordtag_1toN_prob:
            wordtag_1toN_prob[key] /= word_count_1toN[key[0]]
        print('wordtag_1toN_prob size:', len(wordtag_1toN_prob))


        del word_count_1to1
        del word_count_1toN

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
            for tag, score in toptwo:
                wordtag_score[(word, tag)] = score

        # TODO Combine multiple tagged corpora

        # TODO: Evaluate the target tags using annotated corpus.

if __name__ == '__main__':
    main()
