import math

def compute_perplexity(translation_probs):
    return -sum(math.log2(prob) for prob in translation_probs)

def compute_log_likelihood(translation_probs):
    return sum(math.log(prob) for prob in translation_probs)
