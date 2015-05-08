Assignment 1 - IBM-M1, IBM-M2
=============================

Requirements
============

* Python 3

Running
-------

To show all available options:

    $ python3 main.py --help

To run the simple IBM-M1 model:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M1

Number of iterations can be set up:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M1 --iter-1 5

To see all viterbi alignments a debug file can be generated:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M1 --debug debug.txt
 
To generate a NAACL compatible viterbi alignment output:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M1 --output alignments.out

IBM-M2 can be run with an intialization by Model 1 or by random weights:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M2-1
    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M2-Rand

Amount of iterations for the IBM-M2 can also be set:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M2-1 --iter-2 5

Weights can be exported to use them again in the future exploration. The following command will generate two
files weights.q and weights.t with weights:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M2-1 --export weights

Weights can be imported for training:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M2-Rand --import weights

For calculating precision, recall, AER a gold standard can also be set:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --ibm IBM-M2-1 --wa ../data/test.wa.nonullalign

Evaluation
----------

    $ perl EvaluationTools/wa_check_align.pl alignments.out
    $ perl EvaluationTools/wa_eval_align.pl ../data/test.wa.nonullalign alignments.out
