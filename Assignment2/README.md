### Configuring
```
  mkvirtualenv --python=python3 nlp2
  workon nlp2
  pip install -r ../requirements.txt

  ./configure
  python3 create_fastalign_input.py ../data/europarl-v7.cs-en.10000.en ../data/europarl-v7.cs-en.10000.cs > ../data/en-cs-combined10000.txt
  python3 train.py
```

POS Tagging:

```
  python3 run.py ../data/en-cs-eval.txt
```

### Create small evaluation corpus
# Universal dependencies archive retrieved from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/LRT-1478 and extract cs/cs-ud-test.conllu
# testcorpus created using python3 create_testcorpus.py > test.cs and taking first 10000 lines
