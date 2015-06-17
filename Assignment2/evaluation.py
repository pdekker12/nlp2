#!/usr/bin/env python3

import train
import pickle
import string
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.tokenize import word_tokenize
from pos import core_tags, core_tags_without_start
import heapq
from itertools import combinations
from collections import defaultdict
import operator

evaluated_source_languages = ["en","fr","es","de"]
n_languages = len(evaluated_source_languages)
lin_comb_weights = [1/n_languages] * n_languages # Uniform weights

test_corpus_path = "../data/hu-test10000.txt"

def setup_nltk_tagger(trained_params):
    # State set: the universal POS tags
    states = core_tags
    # Output probabilities: probability of observing word given POS tag
    output_probs = trained_params[1]
    # Transition probabilities: probability of observing tag given
    # previous tag
    transition_probs = trained_params[1]
    # Symbols: the target vocabulary from the training corpus
    states = trained_params[2]
    
    return HiddenMarkovModelTagger(states,transition_probs,output_probs,prior_probs)

def run_trained_tagger(output_probs, transition_probs, raw_lines):
    distribution_all_lines = []
    result_all_lines = []
    for line in raw_lines:
        prev_tag = '$'
        distribution = []
        result = []
        for w in line:
            w_score = []
            found = False
            for tag in core_tags_without_start:
                key = (w, tag)
                if key in output_probs:
                    bigram = (prev_tag,tag)
                    if bigram in transition_probs:
                        found = True
                        w_score.append((tag, output_probs[key] * transition_probs[bigram]))
            if found == False:
                w = 'UNK'
                # TODO: Remove this copypaste
                for tag in core_tags_without_start:
                    key = (w, tag)
                    if key in output_probs:
                        bigram = (prev_tag,tag)
                        if bigram in transition_probs:
                            w_score.append((tag, output_probs[key] * transition_probs[bigram]))
            prev_tag = heapq.nlargest(1, w_score, lambda x: x[1])[0][0]
            prev_tag_prob = heapq.nlargest(1, w_score, lambda x: x[1])[0][1]
            distribution.append(dict(w_score)) # add all possible tags and possibiliees
            result.append((prev_tag,prev_tag_prob)) # add pair of best tag and its probability
        distribution_all_lines.append(list(zip(line,distribution)))
        result_all_lines.append(list(zip(line,result)))
    return distribution_all_lines, result_all_lines

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
                    gold_tag = gold_line[j][1]
                    if gold_tag in core_tags_without_start:
                        total_tags +=1
                        word = tagger_line[j][0]
                        if (tagger_line[j][1][0] == gold_tag):
                            correct += 1
                    #else:
                        #print(gold_lines[i])
                        #print("\n")
                        #print(word, ":", tagger_line[j][1][0] , ",", gold_line[j][1])
                        #print("\n")
        accuracy = correct/total_tags
    return accuracy

def linear_combination(distribution, languages):
    first_lang = evaluated_source_languages[0]
    combined_result = []
    # For every parallel line in corpus
    for i in range(len(distribution[first_lang])):
        # For every word in the line
        combined_line = []
        for j in range(len(distribution[first_lang][i])):
            lin_combination = defaultdict(float)
            word = distribution[first_lang][i][j][0]
            # Linearly combine tag probabilties from taggers
            for tag in core_tags_without_start:
                for l in range(len(languages)):
                    lang = languages[l]
                    prob = 0.0
                    if tag in distribution[lang][i][j][1]:
                        prob = distribution[lang][i][j][1][tag]
                    lin_combination[tag] += lin_comb_weights[l] * prob
            # Pick max tag
            max_prob = 0.0
            for tag in core_tags_without_start:
                if (lin_combination[tag] > max_prob):
                    max_prob = lin_combination[tag]
                    best_tag = tag
            combined_line.append((word,(best_tag,max_prob)))
        combined_result.append(combined_line)
    return combined_result

def majority_tag(result, combination):
    first_lang = evaluated_source_languages[0]
    
    combined_result = []
    # For every parallel line in corpus
    for i in range(len(result[first_lang])):
        # For every word in the line
        combined_line = []
        for j in range(len(result[first_lang][i])):
            # For every language
            proposed_tags = defaultdict(list)
            word = result[first_lang][i][j][0]
            for language in combination:
                tag = result[language][i][j][1][0]
                prob = result[language][i][j][1][1]
                # Save probability to dict,
                # length of list also acts as counter of tag occurences
                proposed_tags[tag].append(prob)
            
            # Look for tag which has been chosen the most
            highest_count=0
            highest_prob = 0.0
            best_tag = ""
            for tag in proposed_tags:
                # If current tag has been chosen more
                # than previous encountered tags
                if (len(proposed_tags[tag]) > highest_count):
                    # Current tag is best tag
                    best_tag = tag
                    highest_count = len(proposed_tags[best_tag])
                    highest_prob = max(proposed_tags[best_tag])
                    
                # If tag has been chosen as much as other tags
                elif (len(proposed_tags[tag]) == highest_count):
                    # Compare probabilities
                    if max(proposed_tags[tag]) > highest_prob:
                        best_tag = tag
                        highest_prob = max(proposed_tags[best_tag])
            combined_line.append( (word,(best_tag,highest_prob)) )
        combined_result.append(combined_line)
    return combined_result

def main():
    
    
    # (not used) Setup NLTK tagger using trained parameters
    #nltk_tagger = setup_nltk_tagger(trained_params)
    
    # Load test corpus, on which algorithms can be run
    raw_lines,tagged_lines = load_test_corpus()
    
    best_tags = {}
    distribution = {}
    # Separate languages
    for language in evaluated_source_languages:
        # Load tagger
        pfile = open(language + ".tagger.out","rb")
        trained_params = pickle.load(pfile)
        #sorted_x = sorted(trained_params[0].items(), key=operator.itemgetter(1))
        #print(sorted_x)
        # Run own tagger, using trained parameter on raw corpus
        distribution[language], best_tags[language] = run_trained_tagger(trained_params[0], trained_params[1], raw_lines)
        #print(result[language])
        accuracy = evaluate(best_tags[language], tagged_lines)
        print("Accuracy", language,": ", accuracy)
    
    # Combine languages
    all_combinations = list(map(list,combinations(evaluated_source_languages,2))) + [evaluated_source_languages]
    for combination in all_combinations:
        print("Majority tag of", combination)
        combined_result_maj = majority_tag(best_tags,combination)
        accuracy_maj = evaluate(combined_result_maj, tagged_lines)
        print("Accuracy", combination,": ", accuracy_maj)
        
        print("Linear tag combination of", combination)
        combined_result_lin = linear_combination(distribution,combination)
        accuracy_lin = evaluate(combined_result_lin, tagged_lines)
        print("Accuracy", combination,": ", accuracy_lin)

if __name__ == '__main__':
    main()
