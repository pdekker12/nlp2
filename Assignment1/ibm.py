import random
import math
from collections import defaultdict
import time

MAX_SENTENCE_LENGTH = 100
MAX_DICT_SIZE = 100000

# e_w - word index from the e sentence, f_w - word index from the f sentence
pair_to_int = lambda e_w, f_w: e_w * MAX_DICT_SIZE + f_w

# j - index of the e sentence, i - index of the f sentence, l - len(e), m - len(f)
quadruple_to_int = lambda j, i, l, m:  ((i * MAX_SENTENCE_LENGTH + j) * MAX_SENTENCE_LENGTH + l) * MAX_SENTENCE_LENGTH + m

triple_to_int = lambda i, l, m: (i * MAX_SENTENCE_LENGTH + l) * MAX_SENTENCE_LENGTH + m

# InitModel, used to find heuristic initial parameters for model 1
class InitModel:
    def train(self, foreign_corpus,source_corpus):
        c_e,c_f,c_e_f = init_c_e_f(foreign_corpus,source_corpus)
        
        llr_source_sum = defaultdict(int)
        
        # Compute counts and co-occurence counts of f_w and e_w in parallel sentences
        n_sentences = len(foreign_corpus)
        # For every sentence in the corpus
        for f, e in zip(foreign_corpus, source_corpus):
            #for e_w in e:
                #c_e[e_w] += 1
            #for f_w in f:
                #c_f[f_w] += 1
                #for e_w in e:
                    #c_e_f[(e_w,f_w)] += 1
                
            #total_e_words += len(e)
            #total_f_words += len(f)
            
            # Check occurence of words
            # Use c_e and c_f of dictionaries of possible words
            for e_w in c_e:
                if e_w in e:
                    c_e[e_w] += 1
                    # Co-occurence
                    for f_w in c_f:
                        if (f_w in f):
                            c_e_f[pair_to_int(e_w,f_w)] += 1
            for f_w in c_f:
                if f_w in f:
                    c_f[f_w] += 1
        
        # Compute p(f): take c_f[f_w] and divide by n_sentences
        p_f = {}
        for f_w in c_f:
            p_f[f_w] = float(c_f[f_w]+1)/float(n_sentences+2)
        
        # Compute p(f|e) = p(f,e)/p(e)
        p_f_e = {}
        for (e_w,f_w) in c_e_f:
            if f_w not in p_f_e:
                p_f_e[f_w] = {}
            p_f_e[f_w][e_w] = float(c_e_f[pair_to_int(e_w,f_w)]+1) / float(c_e[e_w] +2)
                
        
        # Compute LLR for every pair (e,f)
        llr = {}
        for (e_w,f_w) in c_e_f:
            if e_w not in llr:
                llr[e_w] = {}
            # The llr score consists of 4 terms, all combinations of
            # e_w and f_w occurring or not occurring.
            # The 4 terms are composed one after eachother.
            
            # e+ f+
            llr[e_w][f_w] = c_e_f[pair_to_int(e_w,f_w)] * math.log(p_f_e[f_w][e_w]/p_f[f_w])
            
            # e+ f-
            c_e_notf = c_e[e_w] - c_e_f[pair_to_int(e_w,f_w)]
            p_notf_e = 1 - p_f_e[f_w][e_w]
            p_notf = 1 - p_f[f_w]
            llr[e_w][f_w] += c_e_notf * math.log(p_notf_e/p_notf)
            
            # e- f+
            c_note_f = c_f[f_w] - c_e_f[pair_to_int(e_w,f_w)]
            c_note = n_sentences - c_e[e_w]
            ## Sum conditional probabilities of all p(f|x) where x != e
            #p_f_note = 0
            #for x in c_e:
                #if ex is not e_w:
                    #p_f_note += p_f_e[f_w][x]
            p_f_note=float(c_note_f+1)/float(c_note+2)
            llr[e_w][f_w] += c_note_f * math.log(p_f_note/p_f[f_w])
            
            # e- f-
            c_note_notf = n_sentences - c_e[e_w] - c_f[f_w] + c_e_f[pair_to_int(e_w,f_w)]
            p_notf_note = float(c_note_notf+1)/float(c_note+2)
            # p_notf has already been calculated
            llr[e_w][f_w] += c_note_notf * math.log(p_notf_note/p_notf)
            
            # Add this llr to the sum for this source sentence
            llr_source_sum[e_w] += llr[e_w][f_w]
        
        max_value = 0
        # Take highest llr source sentence sum
        for value in list(llr_source_sum.values()):
            if value > max_value:
                max_value = value
        denominator = max_value
        # Use this to normalize* all llr's
        # *)not summing to 1 except for llr's from source sentence
        #   where sum originates from
        for e_w in llr:
            for f_w in llr[e_w]:
                # NULL words get target value probabilities
                if (e_w == 0):
                    llr[e_w][f_w] = p_f[f_w]
                # All other get normalized llr
                else:
                    llr[e_w][f_w] = float(llr[e_w][f_w]) / float(denominator)
        
        return llr
                
        
        

class Model1Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.t[pair_to_int(e_w, f_w)] / sum([self.t[pair_to_int(w, f_w)] for w in e])
    
    # compute t without smoothing
    def compute_t(self,count,total_count,ind):
        return count/total_count

class Model1ImprovedSetup:
    
    def __init__(self,option,voc_size=5000,add_n=1,null_weight=3):
        self.t = None
        self.q = None
        
        self.null_weight = null_weight
       
        
        # Parameters needed for smoothing.
        # Add_n value
        self.n = add_n
        # Hypothesized vocabulary size.
        # Is initialized with foreign corpus set size
        self.V = voc_size
        
        # option parameter stores improvement option:
        # 0: Add-N smoothing
        # 1: Heavy NULL
        # 2: Heuristic initialization
        # 3: all improvements
        self.option = option

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.t[pair_to_int(e_w, f_w)] / sum([self.t[pair_to_int(w, f_w)] for w in e])
    
    
    def compute_t(self,count,total_count,index):
        factor = 1
        # Heavy null
        if self.option==1 or self.option==3 or self.option==4:
            # Multiply weights of null words by a factor
            if index == 0:
                factor = self.null_weight
        
        # Add-n smoothing
        if self.option==0 or self.option==3 or self.option==4:
            return factor * ((count + self.n)/(total_count + self.n * self.V))
        else:
            return factor * (count/total_count)

class Model2Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.q[quadruple_to_int(j, i, l, m)] * self.t[pair_to_int(e_w, f_w)] /\
            sum(self.q[quadruple_to_int(w_j, i, l, m)] * self.t[pair_to_int(w, f_w)] for w, w_j in zip(e, range(l)))
    
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
                score += self.t[pair_to_int(w_e, w_f)]
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
            values = [self.t[pair_to_int(w_e, w_f)] for w_e in e]
            alignments.append(values.index(max(values)))

        return alignments

    """
        Length of the foreign_corpus and source_corpus collections
        should be the same
    """
    def train(self, foreign_corpus, source_corpus, clear=False, callback=None, uniform=False):
        if clear:
            print('Resetting weights')
            self.t = {}
            self.q = {}

            # Initialize language model's parameters by random variables
            # from 0 to 1
            normalizing_coef_t = {}
            normalizing_coef_q = {}
            def add(collection, key, val):
                if key in collection:
                    collection[key] += val
                else:
                    collection[key] = 0
            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)
                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        if (f_w, e_w) not in self.t:
                            val = random.random() if not uniform else 0.1
                            add(normalizing_coef_t, e_w, val)
                            self.t[pair_to_int(e_w, f_w)] = val
                        if (j, i, l, m) not in self.q:
                            val = random.random() if not uniform else 0.1
                            add(normalizing_coef_q, triple_to_int(j, l, m), val)
                            self.q[quadruple_to_int(j, i, l, m)] = val

            for item_key, item_val in self.t.items():
                self.t[item_key] = item_val / normalizing_coef_t[item_key // MAX_DICT_SIZE]
            
            for item_key, item_val in self.q.items():
                self.q[item_key] = item_val / normalizing_coef_q[item_key % (MAX_SENTENCE_LENGTH ** 3)]
                

        # A bit bloody hack to link t and q again
        self.model_setup.t = self.t
        self.model_setup.q = self.q

        for t in range(self.num_iter):
            print('Timestamp:', time.time())
            # Set all count c(...) = 0
            # c_e: e word occurence
            # c_e_f: e and f words occurence at the same time
            c_e,_,c_e_f = init_c_e_f(foreign_corpus,source_corpus)
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
                        c_e_f[pair_to_int(e_w, f_w)] += delta
                        c_e[e_w] += delta
                        c_ji_l_m[quadruple_to_int(j, i, l, m)] += delta
                        c_i_l_m[triple_to_int(i, l, m)] += delta

            # Update LM and alignment probs
            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)

                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        # Compute t based on count and total,
                        # smoothing is dependent on model
                        self.t[pair_to_int(f_w, e_w)] = self.model_setup.compute_t(c_e_f[pair_to_int(e_w, f_w)],c_e[e_w],j)
                        self.q[quadruple_to_int(j, i, l, m)] = c_ji_l_m[quadruple_to_int(j, i, l, m)] / c_i_l_m[triple_to_int(i, l, m)]

            if callback != None:
                callback(self)


def init_c_e_f(foreign_corpus,source_corpus):
    c_e={}
    c_f={}
    c_e_f = {}
    for f, e in zip(foreign_corpus, source_corpus):
        for e_w in e:
            c_e[e_w] = 0
            for f_w in f:
                c_f[f_w] = 0
                c_e_f[pair_to_int(e_w, f_w)] = 0
    return c_e,c_f,c_e_f


def init_c_ji_l_m(foreign_corpus,source_corpus):
    c_i_l_m = {}
    c_ji_l_m = {}
    for f, e in zip(foreign_corpus, source_corpus):
        m = len(f)
        l = len(e)
        for i in range(m):
            c_i_l_m[triple_to_int(i, l, m)] = 0
            for j in range(l):
                c_ji_l_m[quadruple_to_int(j, i, l, m)] = 0
    return c_i_l_m, c_ji_l_m
