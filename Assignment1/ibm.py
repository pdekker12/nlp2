import random
from evaluation import compute_perplexity, compute_log_likelihood
import math
import numpy as np
from collections import defaultdict

# InitModel, used to find heuristic initial parameters for model 1
class InitModel:
    def train(self, foreign_corpus,source_corpus):
        c_e = defaultdict(int)
        c_f = defaultdict(int)
        c_e_f = init_c_e_f(foreign_corpus,source_corpus)
        
        # Compute co-occurence counts of f_w and e_w in parallel sentences
        total_f_words = 0
        for f, e in zip(foreign_corpus, source_corpus):
            for f_w in f:
                for e_w in e:
                    c_e[e_w] += 1 ##?? counted again for every alignment
                    if f_w == e_w:
                        c_e_w += 1
                c_f[f_w] += 1 ##?? counted only once
            total_f_words += len(f)
        
        # Compute p_f: take c_f[f_w] and divide by total_f_words
        p_f = {}
        for f_w in c_f:
            p_f[f_w] = c_f[f_w]/total_f_words
        
        # TODO: Implement p_f_e
        p_f_e = {}
        
        # Compute LLR for every pair (e,f)
        llr = {}
        llr_source_sum = {}
        for (e_w,f_w) in c_e_f:
            if e_w not in llr:
                llr[e_w] = {}
            # + +
            llr[e_w][f_w] = c_e_f[(e_w,f_w)] * log(p_f_e[f_w][e_w]/p_f[f_w])
            # TODO: implement further
            # + -
            # - +
            # - -
            llr_source_sum[e_w] += llr[e_w][f_w]
        
        # Take highest llr source sentence sum
        denominator = np.amax(llr_source_sum.values())
        
        # Use this to normalize* all llr's
        # *)not summing to 1 except for llr's from source sentence
        #   where sum originates from
        for e_w in llr:
            for f_w in llr[e_w]:
                llr[e_w][f_w] = float(llr[e_w][f_w]) / float(denominator)
        
        return llr
                
        
        

class Model1Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.t[(f_w, e_w)] / sum([self.t[(f_w, w)] for w in e])
    
    # compute t without smoothing
    def compute_t(self,count,total_count,ind):
        return count/total_count

class Model1ImprovedSetup:
    
    def __init__(self,option,foreign_voc_size=5000,n=1,null_weight=3):
        self.t = None
        self.q = None
        
        self.null_weight = null_weight
       
        
        # Parameters needed for smoothing.
        # Add_n value
        self.n = n
        # Hypothesized vocabulary size.
        # Is initialized with foreign corpus set size
        self.V = foreign_voc_size
        
        # option parameter stores improvement option:
        # 0: Add-N smoothing
        # 1: Heavy NULL
        # 2: Heuristic initialization
        self.option = option

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.t[(f_w, e_w)] / sum([self.t[(f_w, w)] for w in e])
    
    
    def compute_t(self,count,total_count,index):
        if self.option==0:
            return (count + self.n)/(total_count + self.n * self.V)
        elif self.option==1:
            # Multiply weights of null words by a factor
            if index == 0:
                return self.null_weight * (count/total_count)
            else:
                # Normal formula for other words
                return count/total_count
        else:
            return count/total_count

class Model2Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.q[(j, i, l, m)] * self.t[(f_w, e_w)] /\
            sum(self.q[(w_j, i, l, m)] * self.t[(f_w, w)] for w, w_j in zip(e, range(l)))
    
    # compute t without smoothing
    def compute_t(self,count,total_count,ind):
        return count/total_count


class Model:
    """
        IBM Model initial parameters:
        * num_iter - number of training iterations
    """
    def __init__(self, t={}, q={}, model_setup=Model1Setup(), num_iter=3):
        self.num_iter = num_iter

        # TODO: Optimize
        self.t = t
        self.q = q
        self.model_setup = model_setup
        self.model_setup.t = self.t
        self.model_setup.q = self.q

    """
        Calculates a translation score between a pair of sentences f and e
        * f - foreign sentence
        * e - source sentence

        Each sentence is a collection of indexes of words from the dictionary
    """
    def translation_score_normalized(self, f, e):
        return self.translation_prob(f, e) / (len(f) ** len(e))

    def translation_prob(self, f, e):
        score = 0
        for w_f in f:
            for w_e in e:
                score += self.t[(w_f, w_e)]
        return score

    """
        Calculates a viterbi alignment between a pair of sentences f and e
        * f - foreign sentence
        * e - source sentence

        Each sentence is a collection of indexes of words from the dictionary
    """
    def align_viterbi(self, f, e):
        alignments = []
        for w_f, i in zip(f, range(len(f))):
            values = [self.t[(w_f, w_e)] for w_e in e]
            alignments.append(values.index(max(values)))

        return alignments

    """
        Length of the foreign_corpus and source_corpus collections
        should be the same
    """
    def train(self, foreign_corpus, source_corpus, clear=False, callback=None):
        if clear:
            print('Resetting weights')
            self.t = {}
            self.q = {}

            # Initialize language model's parameters by random variables
            # from 0 to 1
            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)
                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        if (f_w, e_w) not in self.t:
                            self.t[(f_w, e_w)] = random.random()
                        if (j, i, l, m) not in self.q:
                            self.q[(j, i, l, m)] = random.random()

        # A bit bloody hack to link t and q again
        self.model_setup.t = self.t
        self.model_setup.q = self.q

        for t in range(self.num_iter):
           

            # Set all count c(...) = 0
            # c_e: e word occurence
            # c_e_f: e and f words occurence at the same time
            c_e,c_e_f = init_c_e_f(foreign_corpus,source_corpus)
            # c_i_l_m: alignment from i (Fra) occurence
            # c_ji_l_m: alignment from j <- i (Eng <- Fra) occurence
            c_i_l_m,c_ji_l_m = init_c_ji_l_m(foreign_corpus,source_corpus)

            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)

                # A loop for words
                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        delta = self.model_setup.delta(f_w, i, e_w, j, e, l, m)
                        c_e_f[(e_w, f_w)] += delta
                        c_e[e_w] += delta
                        c_ji_l_m[(j, i, l, m)] += delta
                        c_i_l_m[(i, l, m)] += delta

            # Update LM and alignment probs
            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)

                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        # Compute t based on count and total,
                        # smoothing is dependent on model
                        self.t[(f_w, e_w)] = self.model_setup.compute_t(c_e_f[(e_w, f_w)],c_e[e_w],j)
                        self.q[(j, i, l, m)] = c_ji_l_m[(j, i, l, m)] / c_i_l_m[(i, l, m)]

            if callback != None:
                callback(self)


def init_c_e_f(foreign_corpus,source_corpus):
    c_e={}
    c_e_f = {}
    for f, e in zip(foreign_corpus, source_corpus):
        for e_w in e:
            c_e[e_w] = 0
            for f_w in f:
                c_e_f[(e_w, f_w)] = 0
    return c_e,c_e_f


def init_c_ji_l_m(foreign_corpus,source_corpus):
    c_i_l_m = {}
    c_ji_l_m = {}
    for f, e in zip(foreign_corpus, source_corpus):
        m = len(f)
        l = len(e)
        for i in range(m):
            c_i_l_m[(i, l, m)] = 0
            for j in range(l):
                c_ji_l_m[(j, i, l, m)] = 0
    return c_i_l_m, c_ji_l_m
