#!/usr/bin/env python2
# Combines the source and target files of a parallel corpus
# in order to create an input file for fast align
# Usage: create_fastalign_input.py SOURCE_PATH TARGET_PATH

import sys

if len(sys.argv) < 3:
    print "Too few arguments"
else:
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    source_file = open(source_path,"r")
    target_file = open(target_path,"r")
    source_lines = source_file.readlines()
    target_lines = target_file.readlines()
    if (len(source_lines) != len(target_lines)):
        print "Not the same length"
    else:
        for i in range(len(source_lines)):
            source_sentence = source_lines[i][:-1]
            target_sentence = target_lines[i][:-1]
            if (source_sentence != "") and (target_sentence != ""):
                print source_sentence + " ||| " + target_sentence
