#!/usr/bin/env bash

date '+%X'

python3 -u main.py --foreign ../data/combination5000.f --source ../data/combination5000.e --wa ../data/test.wa.nonullalign --ibm IBM-M2-Uniform --iter-2 40 --export weights-ibm-2-uniform

date '+%X'
