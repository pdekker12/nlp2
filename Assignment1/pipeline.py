
from corpus import read_corpus

# Read data
parallel_sentences = read_corpus("data/corpus_1000.nl","data/corpus_1000.en")

# IBM Model 1
ibm1 = train_ibm1(parallel_sentences, 10000)
