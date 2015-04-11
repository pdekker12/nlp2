import random
from evaluation import compute_perplexity

class Model1Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.t[(f_w, e_w)] / sum([self.t[(f_w, w)] for w in e])

class Model2Setup:
    def __init__(self):
        self.t = None
        self.q = None

    def delta(self, f_w, i, e_w, j, e, l, m):
        return self.q[(j, i, l, m)] * self.t[(f_w, e_w)] /\
            sum([self.q[(w_j, i, l, m)] * self.t[(f_w, w)] for w, w_j in zip(e, range(l))])


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

    def _translation_score(self, f, e):
        score = 0
        for w_f in f:
            for w_e in e:
                score += self.t[(w_f, w_e)]

        return score / (len(f) ** len(e))

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
                        self.t[(f_w, e_w)] = c_e_f[(e_w, f_w)] / c_e[e_w]
                        self.q[(j, i, l, m)] = c_ji_l_m[(j, i, l, m)] / c_i_l_m[(i, l, m)]

            perplexity = compute_perplexity([self._translation_score(f, e)
                                             for f, e in zip(foreign_corpus, source_corpus)])
            print(perplexity)
