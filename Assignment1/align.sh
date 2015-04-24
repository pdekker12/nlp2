#!/usr/bin/env bash
#
# $2 - Dutch
# $3 - English
#

if [ $1 -eq 1 ]
then
  python3 main.py --foreign $3 --source $2 --output alignments.out --ibm IBM-M1 --iter-1 3
else
  python3 main.py --foreign $3 --source $2 --output alignments.out --ibm IBM-M2-1 --iter-2 3
fi
