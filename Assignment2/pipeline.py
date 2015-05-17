#!/usr/bin/env python2

import sys
import subprocess
import os

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

corpus_paths = ["../data/en-cs-combined10000.txt"]

def main():
    # Previous steps done by other programs
    # Load one/multiple parallel corpora
    # Align every parallel corpus


    for corpus_path in corpus_paths:
        # Perform alignment
        command = "./fast_align/fast_align"
        output = subprocess.check_output([command, "-i", corpus_path])
        print "output"
        alignment_list = output.split("\n")[:-1]


        # Open corpus
        corpus_file = open(corpus_path,"r")
        corpus = corpus_file.readlines()
        print len(corpus)
        for i in range(len(corpus)):
            split_line = corpus[i].split(" ||| ")
            source_line = split_line[0]
            target_line = split_line[1]
            # POS tag this source line
            source_words=source_line.split()
            target_words=target_line.split()
            print source_words
            source_tags = pos_tag(source_words)
            print source_tags

            # Map source POS tags to target POS tags using alignment.
            # TODO: Combine multiple tagged corpora. Use smoothing.
            # Get alignments for this line
            alignments = alignment_list[i].split()
            target_tags = [None] * len(target_words)
            for a in alignments:
                pair = a.split("-")
                source_ind = int(pair[0])
                target_ind = int(pair[1])
                # Map aligned source tag to target word
                target_tags[target_ind] = (target_words[target_ind],source_tags[source_ind][1])
            print target_words
            print target_tags



        # Evaluate the target tags using annotated corpus.

if __name__=="__main__":
    main()
