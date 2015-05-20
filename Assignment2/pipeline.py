#!/usr/bin/env python3

import locale
import sys
import subprocess
import os

from nltk.tag.stanford import POSTagger

corpus_paths = ['../data/en-cs-combined10000.txt']
encoding = locale.getdefaultlocale()[1]

def main():
    # Previous steps done by other programs
    # Load one/multiple parallel corpora
    # Align every parallel corpus

    # Dictionary which contains the tagged target corpora.
    # Every key is a different source corpus.
    tagged_target = [[]] * len(corpus_paths)
    print('Creating a POS tagger')
    tagger = POSTagger('stanford-postagger-2015-04-20/models/english-bidirectional-distsim.tagger',
                       'stanford-postagger-2015-04-20/stanford-postagger.jar')
    print('POS tagger created!')

    for corpus_path in corpus_paths:
        corpus_path_id = corpus_paths.index(corpus_path)

        # Perform alignment
        command = './fast_align/fast_align'
        output = subprocess.check_output([command, '-i', corpus_path])
        alignment_list = output.decode(encoding).split('\n')[:-1]
        print('Got alignments:', len(alignment_list))

        print('Openning the corpus')
        # Open corpus
        corpus_file = open(corpus_path, 'r')
        corpus = corpus_file.readlines()
        for corpus_line, alignment in zip(corpus, alignment_list):
            source_line, target_line = tuple(corpus_line.split(' ||| '))

            # POS tag this source line
            source_words = source_line.split()
            target_words = target_line.split()
            source_tags = tagger.tag(source_words)[0]

            # Map source POS tags to target POS tags using alignment.
            # TODO: Use smoothing.
            # Get alignments for this line
            alignments = alignment.split()
            target_tags = [None] * len(target_words)
            for a in alignments:
                source_ind, target_ind = tuple(map(int, a.split('-')))
                # Map aligned source tag to target word
                target_tags[target_ind] = (target_words[target_ind], source_tags[source_ind][1])
            print(target_tags)
            tagged_target[corpus_path_id].append(target_tags)

        # TODO Combine multiple tagged corpora

        # TODO: Evaluate the target tags using annotated corpus.

if __name__ == '__main__':
    main()
