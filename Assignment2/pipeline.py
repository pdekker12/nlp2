from nltk.tag import pos_tag 
from nltk.tokenize import word_tokenize
import sys
import subprocess
import os

corpus_paths = ["data/en-cs-combined10000.txt"]

def main():
    # Previous steps done by other programs
        # Load one/multiple parallel corpora
        # Align every parallel corpus
    
    # Dictionary which contains the tagged target corpora.
    # Every key is a different source corpus.
    tagged_target = {}
    
    for corpus_path in corpus_paths:
        tagged_target[corpus_path] = []
        
        # Perform alignment
        command = "./fast_align/fast_align"
        output = subprocess.check_output([command, "-i", corpus_path])
        alignment_list = output.split("\n")[:-1]
        
                
        # Open corpus
        corpus_file = open(corpus_path,"r")
        corpus = corpus_file.readlines()
        print len(corpus)
        for i in range(len(corpus)):
            split_line = corpus[i].split(" ||| ")
            source_line = split_line[0]
            target_line = split_line[1]
            # POS tag this source line
            source_words=source_line.split()
            target_words=target_line.split()
            source_tags = pos_tag(source_words)
            
            # Map source POS tags to target POS tags using alignment.
            # TODO: Use smoothing.
            # Get alignments for this line
            alignments = alignment_list[i].split()
            target_tags = [None] * len(target_words)
            for a in alignments:
                pair = a.split("-")
                source_ind = int(pair[0])
                target_ind = int(pair[1])
                # Map aligned source tag to target word
                target_tags[target_ind] = (target_words[target_ind],source_tags[source_ind][1])
            tagged_target[corpus_path].append(target_tags)
        print tagged_target

        # TODO Combine multiple tagged corpora

        # TODO: Evaluate the target tags using annotated corpus.

if __name__=="__main__":
    main()
