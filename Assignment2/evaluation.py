#!/usr/bin/env python3

import train
import pickle
from nltk.tag.hmm import HiddenMarkovModelTagger
from pos import core_tags

test_corpus_path = "../data/cs-test10000.txt"

def setup_nltk_tagger(trained_params):
    # State set: the universal POS tags
    states = core_tags
    # Output probabilities: probability of observing word given POS tag
    output_probs = trained_params[0]
    # Transition probabilities: probability of observing tag given
    # previous tag
    transition_probs = trained_params[1]
    # Symbols: the target vocabulary from the training corpus
    states = trained_params[2]
    
    return HiddenMarkovModelTagger(states,transition_probs,output_probs,prior_probs)

def run_trained_tagger(output_probs, transition_probs, raw_lines):
    result_all_lines = []
    for line in raw_lines:
        prev_tag = '.'
        result = []
        for w in reversed(line):
            w_score = []
            found = False
            for tag in core_tags:
                key = (w, tag)
                if key in output_probs:
                    bigram = (tag, prev_tag)
                    if bigram in transition_probs:
                        found = True
                        w_score.append((tag, output_probs[key] * transition_probs[bigram]))
            if found == False:
                w = 'UNK'
                # TODO: Remove this copypaste
                for tag in core_tags:
                    key = (w, tag)
                    if key in output_probs:
                        bigram = (tag, prev_tag)
                        if bigram in transition_probs:
                            w_score.append((tag, output_probs[key] * transition_probs[bigram]))

            prev_tag = heapq.nlargest(1, w_score, lambda x: x[1])[0][0]
            result.append(prev_tag)
        result_all_lines.append(result[::-1])
    return result

# Load test corpus and convert to list of annotated and unannotated lines
def load_test_corpus():
    unannotated_list = []
    annotated_list = []
    
    corpus_file = open(test_corpus_path,"r")
    lines = corpus_file.readlines()
    for line in lines:
        split_line = word_tokenize(line)
        for token in split_line:
            # Split word and tag
            word,tag = token.split("/")
            # TODO...
            line = [w for w in line if w not in string.punctuation]
            

def main():
    # Load tagger
    pfile = open("tagger.out","rb")
    trained_params = pickle.load(pfile)
    
    # (not used) Setup NLTK tagger using trained parameters
    #nltk_tagger = setup_nltk_tagger(trained_params)
    
    # Load test corpus, on which algorithms can be run
    raw_lines,tagged_lines = load_test_corpus()
    
    # Run own tagger, using trained parameter on unannotated corpus
    result = run_trained_tagger(trained_params[0], trained_params[1], raw_lines)
    

if __name__ == '__main__':
    main()
