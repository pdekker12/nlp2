#!/usr/bin/env python3

import argparse
import pickle
import heapq

from nltk.tokenize import word_tokenize
from pos import core_tags, core_tags_without_start
from itertools import combinations
from collections import Counter, defaultdict

# 0 forward
# 1 backward
# 2 bidirectional

evaluated_source_languages = ["en","fr","es","de"]
n_languages = len(evaluated_source_languages)


test_corpus_path = {"hu": "../data/hu-test10000.txt", "cs": "../data/cs-test10000.txt"}

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


def run_trained_tagger_reverse(output_probs, transition_probs, raw_lines):
    distribution_all_lines = []
    result_all_lines = []
    for line in raw_lines:
        prev_tag = '@'
        distribution = []
        result = []
        for w in reversed(line):
            w_score = []
            found = False
            for tag in core_tags_without_start:
                key = (w, tag)
                if key in output_probs:
                    bigram = (tag, prev_tag)
                    if bigram in transition_probs:
                        found = True
                        w_score.append((tag, output_probs[key] * transition_probs[bigram]))
            if found == False:
                w = 'UNK'
                # TODO: Remove this copypaste
                for tag in core_tags_without_start:
                    key = (w, tag)
                    if key in output_probs:
                        bigram = (tag,prev_tag)
                        if bigram in transition_probs:
                            w_score.append((tag, output_probs[key] * transition_probs[bigram]))
            prev_tag = heapq.nlargest(1, w_score, lambda x: x[1])[0][0]
            prev_tag_prob = heapq.nlargest(1, w_score, lambda x: x[1])[0][1]
            distribution.append(dict(w_score)) # add all possible tags and possibiliees
            result.append((prev_tag,prev_tag_prob)) # add pair of best tag and its probability
        distribution_all_lines.append(list(zip(line,distribution)))
        result_all_lines.append(list(zip(line,result[::-1])))
    return distribution_all_lines, result_all_lines


# Load test corpus and convert to list of tagged and raw lines
def load_test_corpus(tlanguage):
    raw_list = []
    tagged_list = []
    
    corpus_file = open(test_corpus_path[tlanguage],"r")
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
    correct_per_pos = Counter()
    pos_count = Counter()
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
                        pos_count[gold_tag] += 1
                        word = tagger_line[j][0]
                        if (tagger_line[j][1][0] == gold_tag):
                            correct += 1
                            correct_per_pos[gold_tag] += 1
                    #else:
                        #print(gold_lines[i])
                        #print("\n")
                        #print(word, ":", tagger_line[j][1][0] , ",", gold_line[j][1])
                        #print("\n")
        accuracy = correct/total_tags
        for key in correct_per_pos:
            correct_per_pos[key] /= float(pos_count[key])
    return accuracy, correct_per_pos

def linear_combination(distribution, pos_accuracy=None):
    n_languages = len(distribution)
    lin_comb_weights = [1/n_languages] * n_languages # Uniform weights
    #print(lin_comb_weights)
    combined_result = []
    # For every parallel line in corpus
    for i in range(len(distribution[0])):
        # For every word in the line
        combined_line = []
        for j in range(len(distribution[0][i])):
            lin_combination = defaultdict(float)
            word = distribution[0][i][j][0]
            #print(word)
            # Linearly combine tag probabilties from taggers
            for l in range(len(distribution)):
                langresult = distribution[l][i][j]
                #print("Distribution language ",l,langresult)
                for tag in langresult[1]:
                    prob = langresult[1][tag]
                    if pos_accuracy:
                        lin_combination[tag] += pos_accuracy[l][tag] * prob
                    else:
                        lin_combination[tag] += lin_comb_weights[l] * prob
            #print("Linear combination",lin_combination)
            # Pick max tag
            max_prob = 0.0
            for tag in core_tags_without_start:
                if (lin_combination[tag] > max_prob):
                    max_prob = lin_combination[tag]
                    best_tag = tag
            #print("Best tag:",best_tag, max_prob)
            combined_line.append((word,(best_tag,max_prob)))
        combined_result.append(combined_line)
    return combined_result

def majority_tag(result):
    combined_result = []
    # For every parallel line in corpus
    for i in range(len(result[0])):
        # For every word in the line
        combined_line = []
        for j in range(len(result[0][i])):
            # For every language
            proposed_tags = defaultdict(list)
            word = result[0][i][j][0]
            for res in result:
                tag = res[i][j][1][0]
                prob = res[i][j][1][1]
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


def main(args):
    
    DIRECTION = int(args.direction)
    tlanguage = args.target
    # Load test corpus, on which algorithms can be run
    raw_lines,tagged_lines = load_test_corpus(tlanguage)
    
    best_tags = {}
    distribution = {}

    # Separate languages
    separate_language_accuracy = []
    separate_language_pos_accuracy = []
    for slanguage in evaluated_source_languages:
        # Load tagger
        pfile = open(slanguage + "-" + tlanguage + ".tagger.out","rb")
        trained_params = pickle.load(pfile)
        if (DIRECTION==0): # forward
            print("Forward tagging")
            # Run own tagger, using trained parameter on raw corpus
            distribution[slanguage], best_tags[slanguage] = run_trained_tagger(trained_params[0], trained_params[1], raw_lines)
        elif (DIRECTION==1): # backward
            print("Backward tagging")
            distribution[slanguage], best_tags[slanguage] = run_trained_tagger_reverse(trained_params[0], trained_params[1], raw_lines)
        elif (DIRECTION==2): # bidirectional
            print("Bidirectional tagging")
            distribution1, best_tags1 = run_trained_tagger(trained_params[0], trained_params[1], raw_lines)

            distribution2, best_tags2 = run_trained_tagger_reverse(trained_params[0], trained_params[1], raw_lines)
            best_tags[slanguage] = majority_tag([best_tags1,best_tags2]) # combine two directions
            
        accuracy, accuracy_per_pos = evaluate(best_tags[slanguage], tagged_lines)
        separate_language_accuracy.append(accuracy)
        separate_language_pos_accuracy.append(accuracy_per_pos)
        print( slanguage,"&", accuracy)

    norm = sum(separate_language_accuracy)
    separate_language_accuracy = map(lambda x: x / norm, separate_language_accuracy)
    if args.weight_acc:
        lin_comb_weights = list(separate_language_accuracy)
        print('New weights:', lin_comb_weights)

    if args.weight_pos:
        for tag in core_tags:
            try:
                norm = sum(pos_accuracy[tag] for pos_accuracy in separate_language_pos_accuracy)
                for i in range(len(separate_language_pos_accuracy)):
                    separate_language_pos_accuracy[i][tag] /= norm
            except ZeroDivisionError:
                del separate_language_pos_accuracy[i][tag]
    else:
        separate_language_pos_accuracy = None
    
    # Combine languages
    all_combinations = list(map(list,combinations(evaluated_source_languages,2))) + list(map(list,combinations(evaluated_source_languages,3))) + [evaluated_source_languages]
    for combination in all_combinations:
        results_best_tags = []
        names=""
        for lang in combination:
            results_best_tags.append(best_tags[lang])
            names += lang + "-"
        print("Majority tag of", combination)
        combined_result_maj = majority_tag(results_best_tags)
        accuracy_maj,_ = evaluate(combined_result_maj, tagged_lines)
        print("Accuracy", combination,": ", accuracy_maj)
        if (DIRECTION != 2):
            results_distribution = []
            for lang in combination:
                results_distribution.append(distribution[lang])
            print("Linear tag combination of", combination)
            combined_result_lin = linear_combination(results_distribution, separate_language_pos_accuracy)
            accuracy_lin,_ = evaluate(combined_result_lin, tagged_lines)
            print("Accuracy", combination,": ", accuracy_lin)
        
        print(names,"&&",accuracy_maj,accuracy_lin)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--weight_acc', action='store_true', default=False, help='Accuracy dependent weights')
    parser.add_argument('--weight_pos', action='store_true', default=False, help='POS dependent weights')
    parser.add_argument('--target', default="cs",help='Target language')
    parser.add_argument('--direction', default=0,help='Direction: 0 forward, 1 backward, 2 bidirectional')
    args = parser.parse_args()
    main(args)
