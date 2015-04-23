#!/usr/bin/env bash

date '+%X'

python3 -u main.py --foreign ../data/combination5000.f --source ../data/combination5000.e --wa ../data/test.wa.nonullalign --ibm IBM-M1 --iter-1 40

date '+%X'
