#!/usr/bin/env bash

date '+%X'

python3 -u main.py --foreign ../data/combination5000.f --source ../data/combination5000.e --wa ../data/test.wa.nonullalign --ibm IBM-M2-1 --iter-2 40

date '+%X'
