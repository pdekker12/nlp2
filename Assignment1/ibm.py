import random
from evaluation import compute_perplexity

class Model1Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.t[(f_w, e_w)] / sum([self.t[(f_w, w)] for w in e])
    
    # compute t without smoothing
    def compute_t(self,count,total_count):
        return count/total_count

class Model1ImprovedSetup:
    
    def __init__(self,foreign_voc_size,n):
        self.t = None
        self.q = None
        
        # Parameters needed for smoothing.
        # Add_n value
        self.n = n
        # Hypothesized vocabulary size.
        # Is initialized with foreign corpus set size
        self.V = foreign_voc_size

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.t[(f_w, e_w)] / sum([self.t[(f_w, w)] for w in e])
    
    
    def compute_t(self,count,total_count):
        return (count + self.n)/(total_count + self.n * self.V)

class Model2Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.q[(j, i, l, m)] * self.t[(f_w, e_w)] /\
            sum([self.q[(w_j, i, l, m)] * self.t[(f_w, w)] for w, w_j in zip(e, range(l))])
    
    # compute t without smoothing
    def compute_t(self,count,total_count):
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
    def translation_score(self, f, e):
        score = 0
        for w_f in f:
            for w_e in e:
                score += self.t[(w_f, w_e)]

        return score / (len(f) ** len(e))

    """
        Calculates a viterbi alignment between a pair of sentences f and e
        * f - foreign sentence
        * e - source sentence

        Each sentence is a collection of indexes of words from the dictionary
    """
    def align_viterbi(self, f, e):
        alignments = [[] for i in range(len(e))]
        for w_f, i in zip(f, range(len(f))):
            values = [self.t[(w_f, w_e)] for w_e in e]
            max_index = values.index(max(values))
            alignments[max_index].append(i)

        return alignments

    """
        Length of the foreign_corpus and source_corpus collections
        should be the same
    """
    def train(self, foreign_corpus, source_corpus, clear=False):
        if clear:
            self.t = {}
            self.q = {}

            # A bit blood hack to link t and q again
            self.model_setup.t = self.t
            self.model_setup.q = self.q
            

            # Initialize language model's parameters by random variables
            # from 0 to 1
            for f, e in zip(foreign_corpus, source_corpus):
                for f_w in f:
                    for e_w in e:
                        if (f_w, e_w) not in self.t:
                            self.t[(f_w, e_w)] = random.random()

        for t in range(self.num_iter):
            # e and f words occurence at the same time
            c_e_f = {}

            # e word occurence
            c_e = {}

            # alignment from j <- i (Eng <- Fra) occurence
            c_ji_l_m = {}

            # alignment from i (Fra) occurence
            c_i_l_m = {}

            # Set all count c(...) = 0
            # TODO: Move these initilizations into the separate function
            # to avoid duplicates
            for f, e in zip(foreign_corpus, source_corpus):
                for e_w in e:
                    c_e[e_w] = 0
                    for f_w in f:
                        c_e_f[(e_w, f_w)] = 0

            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)
                for i in range(m):
                    c_i_l_m[(i, l, m)] = 0
                    for j in range(l):
                        c_ji_l_m[(j, i, l, m)] = 0

            #k = 0
            for f, e in zip(foreign_corpus, source_corpus):
                #k += 1
                #print('Sentence', k)
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
                        self.t[(f_w, e_w)] = self.model_setup.compute_t(c_e_f[(e_w, f_w)],c_e[e_w])
                        self.q[(j, i, l, m)] = c_ji_l_m[(j, i, l, m)] / c_i_l_m[(i, l, m)]

            perplexity = compute_perplexity([self.translation_score(f, e)
                                             for f, e in zip(foreign_corpus, source_corpus)])
            print(perplexity)
