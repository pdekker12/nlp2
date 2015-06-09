#!/usr/bin/env python3
# Combines the source and target files of a parallel corpus
# in order to create an input file for fast align
# Usage: create_fastalign_input.py SOURCE_PATH TARGET_PATH

import sys
import string
import os
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
from collections import defaultdict

years = range(2006,2011+1)
months = list(range(1,12+1))
days = range(1,31-1)

languages = ["cs","da","de","el","en","es","et","fi","fr","hu","it","lt","lv","nl","pl","pt","sk","sl","sv"]
language_file = {}
for language in languages:
    path = "../data/europarl/" + language + ".txt"
    print(path)
    language_file[language] = open(path,"w")


#~ if len(sys.argv) < 3:
    #~ print('Too few arguments')
    #~ sys.exit(-1)
#~ 
#~ source_path = sys.argv[1]
#~ target_path = sys.argv[2]
#~ source_file = open(source_path, 'r')
#~ target_file = open(target_path, 'r')

#sentences = defaultdict(list)

for year in years:
    for month in months:
        for day in days:
            date = str(year) + "-" + str(month).zfill(2) + "-" + str(day).zfill(2)
            path = "../data/sessions/" + date + ".xml"
            if os.path.exists(path):
                print(path)
                tree = ET.parse(path)
                session = tree.getroot()
                for chapter in session:
                    for part in chapter:
                        if(part.tag=="turn"):
                            for speaker in part:
                                # First check if all languages have been found
                                found_languages = []
                                number_paragraphs = []
                                for text in speaker:
                                    number_paragraphs.append(len(text))
                                    found_languages.append(text.attrib["language"])
                                
                                # If all languages have been found for this turn...
                                if set(found_languages) == set(languages):
                                    # ...and number of paragraphs is the same for every language, then add
                                    if len(set(number_paragraphs)) == 1:
                                        for text in speaker:
                                            language = text.attrib["language"]
                                            if (language in languages):
                                                for p in text:
                                                    # Add line to corresponding file
                                                    content = str(p.text) +"\n"
                                                    language_file[language].write(content)

for language in languages:
    language_file[language].close()

#for source_sentence, target_sentence in zip(source_file, target_file):
    #source_tokens = word_tokenize(source_sentence)
    #target_tokens = word_tokenize(target_sentence)

    #source_tokens = [w for w in source_tokens if w not in string.punctuation]
    #target_tokens = [w for w in target_tokens if w not in string.punctuation]
    #if len(source_tokens) > 0 and len(target_tokens) > 0:
        #source_sentence = ' '.join(source_tokens)
        #target_sentence = ' '.join(target_tokens)
        #print(source_sentence + ' ||| ' + target_sentence)
