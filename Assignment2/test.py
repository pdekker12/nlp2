#!/usr/bin/env python3

import train
import unittest

class PipelineChecker(unittest.TestCase):
    def test_pipeline(self):
        train.corpus_paths = ['../data/en-cs-test.txt']
        train.main()

if __name__ == '__main__':
    unittest.main()
