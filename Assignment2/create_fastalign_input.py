#!/usr/bin/env python3
# Combines the source and target files of a parallel corpus
# in order to create an input file for fast align
# Usage: create_fastalign_input.py SOURCE_PATH TARGET_PATH

import sys
import string
from nltk.tokenize import word_tokenize

if len(sys.argv) < 3:
    print('Too few arguments')
    sys.exit(-1)

source_path = sys.argv[1]
target_path = sys.argv[2]
source_file = open(source_path, 'r')
target_file = open(target_path, 'r')

for source_sentence, target_sentence in zip(source_file, target_file):
    source_tokens = word_tokenize(source_sentence)
    target_tokens = word_tokenize(target_sentence)

    source_tokens = [w for w in source_tokens if w not in string.punctuation]
    target_tokens = [w for w in target_tokens if w not in string.punctuation]
    if len(source_tokens) > 0 and len(target_tokens) > 0:
        source_sentence = ' '.join(source_tokens)
        target_sentence = ' '.join(target_tokens)
        print(source_sentence + ' ||| ' + target_sentence)
