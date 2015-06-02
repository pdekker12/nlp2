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
