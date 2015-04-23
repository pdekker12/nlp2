#!/usr/bin/env bash

date '+%X'

python3 -u main.py --foreign ../data/combination_5447.f --source ../data/combination_5447.e --wa ../data/test.wa.nonullalign --ibm IBM-M2-Rand --iter-2 60

date '+%X'
