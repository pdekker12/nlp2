#!/usr/bin/env bash

if [ $1 -eq 1 ]
then
  python3 main.py --foreign $3 --source $2 --debug alignments.out --ibm IBM-M1
else
  # TODO: Use IBM-M2 with random initialization or with parameters from IBM-M1 ?
  # TODO: Check if it's a required output format
  python3 main.py --foreign $3 --source $2 --debug alignments.out --ibm IBM-M2-1
fi
