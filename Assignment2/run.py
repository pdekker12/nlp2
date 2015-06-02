#!/usr/bin/env python3
import pickle
import sys
import string
import heapq
from pos import core_tags
from nltk.tokenize import word_tokenize


if __name__ == '__main__':
    score, npos_count = pickle.load(open("tagger.out", "rb"))
    with open(sys.argv[1]) as input_file:
        for line in input_file:
            line = word_tokenize(line)
            line = [w for w in line if w not in string.punctuation]
            print(line)
            prev_tag = '.'
            result = []
            for w in reversed(line):
                w_score = []
                found = False
                for tag in core_tags:
                    key = (w, tag)
                    if key in score:
                        bigram = (tag, prev_tag)
                        if bigram in npos_count:
                            found = True
                            w_score.append((tag, score[key] * npos_count[bigram]))
                if found == False:
                    w = 'UNK'
                    # TODO: Remove this copypaste
                    for tag in core_tags:
                        key = (w, tag)
                        if key in score:
                            bigram = (tag, prev_tag)
                            if bigram in npos_count:
                                w_score.append((tag, score[key] * npos_count[bigram]))

                prev_tag = heapq.nlargest(1, w_score, lambda x: x[1])[0][0]
                result.append(prev_tag)
            print(result[::-1])

