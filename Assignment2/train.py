#!/usr/bin/env python3

import subprocess
import heapq
import pickle
import argparse
from pos import generic_to_core_pos, core_tags
from config import source_languages, corpus_path, tagger_path, chunk_size, encoding
from collections import Counter
from collections import defaultdict

from nltk.tag.stanford import POSTagger

def parse_corpus(corpus_file, language, tagger):
    """
        Parses source and target sentences into lexemes, determines POS of the source
        sentence by chunk of chunk_size sentences per time and yields the triple as a
        tuple.
    """
    source_queue = []
    target_queue = []

    for iteration, corpus_line in enumerate(corpus_file):
        source_line, target_line = corpus_line.split(' ||| ')

        # POS tag this source line
        source_words = source_line.split()
        target_words = target_line.split()
        source_queue.append(source_words)
        target_queue.append(target_words)

        if iteration % chunk_size == chunk_size - 1:
            try:
                source_tags = tagger.tag_sents(source_queue)
                for source, target, tags in zip(source_queue, target_queue, source_tags):
                    yield source, target, list(map(lambda tag: (tag[0], generic_to_core_pos(language, tag[1])), tags))
            except UnicodeDecodeError:
                for i in range(chunk_size):
                    yield None, None, None
            source_queue = []
            target_queue = []

    if source_queue:
        source_tags = tagger.tag_sents(source_queue)
        for source, target, tags in zip(source_queue, target_queue, source_tags):
            yield source, target, list(map(lambda tag: (tag[0], generic_to_core_pos(language, tag[1])), tags))


def mt_alignment(corpus_path):
    """
        Yields a list of alignments for each sentence.
    """
    command = './fast_align/fast_align'
    output = subprocess.check_output([command, '-I 20', '-i', corpus_path])
    alignment_list = output.decode(encoding).split('\n')[:-1]
    del output

    for alignment in alignment_list:
        yield [tuple(map(int, a.split('-'))) for a in alignment.split()]


def create_stanford_postagger(tagger_path):
    return POSTagger(tagger_path, 'stanford-postagger-full-2015-04-20/stanford-postagger.jar')


def wordtag_score(wordtag_1to1_prob, wordtag_1toN_prob):
    """
        Linear combines two 1to1 and 1toN estimators to receive P(t_i|w_i).
        
        Fossum & Abney paper, page 4.
    """
    pos_score = {}

    uniq_keys = set(wordtag_1to1_prob.keys()).union(wordtag_1toN_prob.keys())
    for key in uniq_keys:
        if key in wordtag_1to1_prob:
            score = 0.5 * wordtag_1to1_prob[key]
            if key in wordtag_1toN_prob:
                score += 0.5 * wordtag_1toN_prob[key]
            pos_score[key] = score
        else:
            pos_score[key] = wordtag_1toN_prob[key]

    return pos_score


def add_unk(wordtag_1to1_prob, wordtag_1toN_prob, word_count):
    """
        Adds an unknown word
        Yeah, a bit bullsh*t, O(n) again. Maybe should be optimized later.
    """
    # Words that occur once
    rare_words = {word for word, count in word_count.items() if count == 1}
    for word in rare_words:
        del word_count[word]
    word_count['UNK'] = len(rare_words)

    def update_wordtag_prob(wordtag_prob):
        rare_word_tags = [(word, tag, score) for (word, tag), score in wordtag_prob.items() if word in rare_words]
        unk_prob = Counter()
        for word, tag, score in rare_word_tags:
            unk_prob[tag] += score
            del wordtag_prob[(word, tag)]

        for tag, score in unk_prob.items():
            wordtag_prob[('UNK', tag)] = score

    update_wordtag_prob(wordtag_1to1_prob)
    update_wordtag_prob(wordtag_1toN_prob)


def normalize(dictionary):
    norm = sum(dictionary.values())
    for key in dictionary:
        dictionary[key] /= norm


def corpus_stat(slanguage, tlanguage, tagger):
    """
        Calculates components of the noisy channel equation
    """
    wordtag_1to1_prob = Counter()
    wordtag_1toN_prob = Counter()
    word_count = Counter()
    pos_count = Counter()
    npos_count = Counter()
    all_target_tokens = []

    with open(corpus_path[tlanguage][slanguage], 'r') as corpus_file:
        for (source_words, target_words, source_tags), alignments in zip(parse_corpus(corpus_file, slanguage, tagger),
            if not source_words:
                continue

            all_target_tokens += target_words
            link_count = [0] * len(source_words)
            for source_id, _ in alignments:
                link_count[source_id] += 1
            one_to_n_marker = [count > 1 for count in link_count]

            tag_seq = ['$']
            for (source_ind, _), target_word in zip(alignments, target_words):
                pos_tag = source_tags[source_ind][1]
                tag_seq.append(pos_tag)
                key = (target_word, pos_tag)
                word_count[target_word] += 1
                pos_count[pos_tag] += 1
                if one_to_n_marker[source_ind]:
                    wordtag_1toN_prob[key] += 1
                else:
                    wordtag_1to1_prob[key] += 1
            tag_seq.append('@')
            for tag1, tag2 in zip(tag_seq, tag_seq[1:]):
                npos_count[(tag1, tag2)] += 1
    add_unk(wordtag_1to1_prob, wordtag_1toN_prob, word_count)
    
    target_vocabulary = set(all_target_tokens)
    
    # Normalizing counters
    normalize(wordtag_1to1_prob)
    normalize(wordtag_1toN_prob)

    return wordtag_score(wordtag_1to1_prob, wordtag_1toN_prob), word_count, pos_count, npos_count, target_vocabulary


def noisy_channel_params(slanguage, tlanguage, tagger):
    """
        Magic with smoothing...
    """
    word_given_tag_score, word_prob, pos_prob, npos_count, target_vocabulary = corpus_stat(slanguage, tlanguage, tagger)

    word_to_tags = defaultdict(list)
    for (word, tag), score in word_given_tag_score.items():
        word_to_tags[word].append((tag, score))

    word_given_tag_score = {}
    for word, core_tags in word_to_tags.items():
        toptwo_cores = heapq.nlargest(2, core_tags, lambda x: x[1])
        for tag, score in toptwo_cores:
            word_given_tag_score[(word, tag)] = score

    normalize(word_given_tag_score)

    return word_given_tag_score, pos_prob, word_prob, npos_count, target_vocabulary


def pos_score(slanguage, tlanguage, tagger):
    """
        Calculates P(w_i|t_i)
        p(t_i|w_i) = c(w_i, t_i) / c(t_i)
        p(t_i) = c(t_i) / c
        p(w_i) = c(w_i) / c
        p(w_i|t_i) = p(t_i|w_i) * c(w_i) / c(t_i)
    """
    wordtag_score, pos_count, word_count, npos_count, target_vocabulary = noisy_channel_params(slanguage, tlanguage, tagger)
    score = {(word, tag): score * word_count[word] / pos_count[tag] for (word, tag), score in wordtag_score.items()}
    return score, npos_count, target_vocabulary


def smooth_wb(npos_count):
    # Compute N,T and Z counters, needed for smoothing
    tags_after = defaultdict(list)
    for tag1 in core_tags:
        for tag2 in core_tags:
            if (tag1, tag2) in npos_count:
                tags_after[tag1].append((tag2, npos_count[(tag1, tag2)]))
    
    # N is the number of tag tokens encountered after tag1
    N = Counter()
    for tag1 in tags_after:
        for tag2,count in tags_after[tag1]:
            N[tag1] += count
    
    # N_total is the total number of tags
    N_total = sum(N.values())
    
    # T is the number of tag types encountered after tag1
    T = Counter()
    for tag1 in tags_after:
        T[tag1] = len(tags_after[tag1])

    # Z is the number of tag types after tag1 not encountered in the training data
    Z = Counter()
    for tag1 in T:
        for tag2 in core_tags:
            if tag2 not in tags_after[tag1]:
                Z[tag1] += 1

    transition_prob = {}
    # Create dict of smoothed probabilities
    # for all combinations of core tags
    for tag1 in core_tags:
        for tag2 in core_tags:
            # Check if (tag1,tag2) has been found and has count > 0
            if ((tag1, tag2) in npos_count) and npos_count[(tag1, tag2)] > 0:
                # If count > 0, use this count to compute smoothed probability
                transition_prob[(tag1, tag2)] = npos_count[(tag1, tag2)] / (N[tag1] + T[tag1])
            else:
                # If count == 0, use T to compute smoothed probability
                transition_prob[(tag1, tag2)] = T[tag1] / (Z[tag1] + (N_total + T[tag1]))

    return transition_prob


def main(args):
    tlanguage = args.target
    for slanguage in source_languages:
        print('Creating a POS tagger for', slanguage)
        tagger = create_stanford_postagger(tagger_path[slanguage])
        print('POS tagger created!')

        score, npos_count, target_vocabulary = pos_score(slanguage, tlanguage, tagger)
        transition_probs = smooth_wb(npos_count)
        print(transition_probs)
        pickle.dump((score, transition_probs,target_vocabulary), open(slanguage + '-' + tlanguage + '.tagger.out', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--target', default="cs", help='Target language')
    args = parser.parse_args()
    main(args)
