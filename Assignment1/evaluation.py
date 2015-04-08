
import numpy as

def compute_perplexity(t):
    # Sum log_2 probabilities for word pairs in t
    sum = 0.0
    for pair in t:
        sum += np.log2(t[pair])
    return -sum
