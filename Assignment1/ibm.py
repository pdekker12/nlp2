import random

class Model1:
    """
        IBM Model 1 initial parameters:
        * num_iter - number of training iterations
    """
    def __init__(self, num_iter=3):
        self.num_iter = num_iter

        # TODO: Optimize
        self.t = {}
        self.q = {}

    """
        Length of the foreign_corpus and source_corpus collections
        should be the same
    """
    def train(self, foreign_corpus, source_corpus):
        self.t = {}
        self.q = {}

        # Initialize language model's parameters by random variables
        # from 0 to 1
        for f, e in zip(foreign_corpus, source_corpus):
            for f_w in f:
                for e_w in e:
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

            k = 0
            for f, e in zip(foreign_corpus, source_corpus):
                k += 1
                print('Sentence', k)
                m = len(f)
                l = len(e)

                # A loop for words
                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        delta = self.t[(f_w, e_w)] / sum([self.t[(f_w, w)] for w in e])
                        # TODO: Remove duplicate
                        c_e_f[(e_w, f_w)] += delta
                        c_e[e_w] += delta
                        c_ji_l_m[(j, i, l, m)] += delta
                        c_i_l_m[(i, l, m)] += delta

            # Update LM and alignment probs
            # TODO: Remove duplicate
            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)

                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        self.t[(f_w, e_w)] = c_e_f[(e_w, f_w)] / c_e[e_w]
                        self.q[(j, i, l, m)] = c_ji_l_m[(j, i, l, m)] / c_i_l_m[(i, l, m)]


class Model2:
    """
        IBM Model 2 parameters are initialized by an output
        of IBM Model 1
    """
    def __init__(self, t, q, num_iter=3):
        self.t = t
        self.q = q
        self.num_iter = num_iter

    def train(self, foreign_corpus, source_corpus):
        for s in range(self.num_iter):
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

            k = 0
            for f, e in zip(foreign_corpus, source_corpus):
                k += 1
                print('Sentence', k)
                m = len(f)
                l = len(e)

                # A loop for words
                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        delta = self.q[(j, i, l, m)] * self.t[(f_w, e_w)] /\
                                sum([self.q[(w_j, i, l, m)] * self.t[(f_w, w)] for w, w_j in zip(e, range(l))])
                        c_e_f[(e_w, f_w)] += delta
                        c_e[e_w] += delta
                        c_ji_l_m[(j, i, l, m)] += delta
                        c_i_l_m[(i, l, m)] += delta

            # Update LM and alignment probs
            # TODO: Remove duplicate
            for f, e in zip(foreign_corpus, source_corpus):
                m = len(f)
                l = len(e)

                for f_w, i in zip(f, range(m)):
                    for e_w, j in zip(e, range(l)):
                        self.t[(f_w, e_w)] = c_e_f[(e_w, f_w)] / c_e[e_w]
                        self.q[(j, i, l, m)] = c_ji_l_m[(j, i, l, m)] / c_i_l_m[(i, l, m)]