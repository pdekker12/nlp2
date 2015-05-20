#!/usr/bin/env python3

import pipeline
import unittest

class PipelineChecker(unittest.TestCase):
    def test_pipeline(self):
        pipeline.corpus_paths = ['../data/en-cs-test.txt']
        pipeline.main()

if __name__ == '__main__':
    unittest.main()
