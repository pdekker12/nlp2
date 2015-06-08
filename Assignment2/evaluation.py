#!/usr/bin/env python3

import train
import pickle
import string
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.tokenize import word_tokenize
from pos import core_tags
import heapq


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
        result_all_lines.append(list(zip(line,result[::-1])))
    return result_all_lines

# Load test corpus and convert to list of tagged and raw lines
def load_test_corpus():
    raw_list = []
    tagged_list = []
    
    corpus_file = open(test_corpus_path,"r")
    lines = corpus_file.readlines()
    for line in lines:
        split_line = word_tokenize(line)
        raw_sentence = []
        tagged_sentence = []
        for token in split_line:
            # Split word and tag
            split_token = token.split("\\")
            if len(split_token) > 1:
                word,tag=split_token
                #if word not in string.punctuation:
                raw_sentence.append(word)
                tagged_sentence.append((word,tag))
        raw_list.append(raw_sentence)
        tagged_list.append(tagged_sentence)
    
    return raw_list, tagged_list

def evaluate(tagger_result, gold_lines):
    accuracy = 0
    if (len(tagger_result) != len(gold_lines)):
        print("length different")
    else:
        total_tags = 0
        correct = 0
        for i in range(len(tagger_result)):
            tagger_line = tagger_result[i]
            gold_line = gold_lines[i]
            if (len(tagger_line) != len(gold_line)):
                print("length of line different")
            else:
                for j in range(len(tagger_line)):
                    total_tags +=1
                    if (tagger_line[j][1] == gold_line[j][1]):
                        correct += 1
        print(correct)
        print(total_tags)
        accuracy = correct/total_tags
    return accuracy
    
def main():
    # Load tagger
    pfile = open("tagger.out","rb")
    trained_params = pickle.load(pfile)
    
    # (not used) Setup NLTK tagger using trained parameters
    #nltk_tagger = setup_nltk_tagger(trained_params)
    
    # Load test corpus, on which algorithms can be run
    raw_lines,tagged_lines = load_test_corpus()
    
    # Run own tagger, using trained parameter on raw corpus
    result = run_trained_tagger(trained_params[0], trained_params[1], raw_lines)
    accuracy = evaluate(result, tagged_lines)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
    main()
