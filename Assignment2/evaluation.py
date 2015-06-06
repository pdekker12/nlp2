#!/usr/bin/env python3

import train
import pickle
from nltk.tag.hmm import HiddenMarkovModelTagger
from pos import core_tags

test_corpus_path = "../data/cs-test10000.txt"

def setup_tagger(trained_params):
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

def load_test_corpus():
    pass

def main():
    # Load tagger
    pfile = open("tagger.out","rb")
    trained_params = pickle.load(pfile)
    
    # Setup tagger using trained parameters
    tagger = setup_tagger(trained_params)
    
    # Load test corpus, on which
    

if __name__ == '__main__':
    main()
