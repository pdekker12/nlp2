import numpy as np

def compute_perplexity(translation_probs):
    return -sum(np.log2(translation_probs))

def compute_log_likelihood(translation_probs):
    return sum(np.log2(translation_probs))
