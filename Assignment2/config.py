source_languages = ['en', 'de', 'fr', 'es']

corpus_path = {'hu':
                   {'en': '../data/europarl/en-hu10000.txt',
                    'de': '../data/europarl/de-hu12000.txt',
                    'fr': '../data/europarl/fr-hu10000.txt',
                    'es': '../data/europarl/es-hu10000.txt'},
               'cs':
                   {'en': '../data/europarl/en-cs10000.txt',
                    'de': '../data/europarl/de-cs12000.txt',
                    'fr': '../data/europarl/fr-cs10000.txt',
                    'es': '../data/europarl/es-cs10000.txt'}
              }

tagger_path = {'en': 'stanford-postagger-full-2015-04-20/models/english-bidirectional-distsim.tagger',
               'de': 'stanford-postagger-full-2015-04-20/models/german-hgc.tagger',
               'fr': 'stanford-postagger-full-2015-04-20/models/french.tagger',
               'es': 'stanford-postagger-full-2015-04-20/models/spanish-distsim.tagger'}

chunk_size = 1000

import locale

encoding = locale.getdefaultlocale()[1]

test_corpus_path = {'hu': '../data/hu-test10000.txt',
                    'cs': '../data/cs-test10000.txt'
                    }