### Configuring, for the cleaned europarl corpus
```
  mkvirtualenv --python=python3 nlp2
  workon nlp2
  pip install -r ../requirements.txt

  ./configure
  python3 parse_europarl.py
  python3 create_europarl_combinations.py
  
  python3 train.py
```

### For the toy europarl corpus:
```
python3 create_fastalign_input.py ../data/europarl-v7.cs-en.10000.en ../data/europarl-v7.cs-en.10000.cs > ../data/en-cs-combined10000.txt
```

POS Tagging:

```
  python3 run.py ../data/en-cs-eval.txt
```

### Create small evaluation corpus
# Universal dependencies archive retrieved from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/LRT-1478 and extract cs/cs-ud-test.conllu
# testcorpus created using python3 create_testcorpus.py > ../data/test.cs and head --lines=10000 ../data/test.cs > ../data/cs-test10000.txt

