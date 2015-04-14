Assignment 1 - IBM-M1, IBM-M2
=============================

Configuring
-----------

Please download corpus files into this folder before running the application.

Running
-------

To show all available options:

    $ python3 main.py --help

An example of options:

    $ python3 main.py --foreign ../data/test.f --source ../data/test.e --debug debug.out --ibm IBM-M2-1 --wa ../data/test.wa.nonullalign --iter-2 5 --output alignments.out

Evaluation
----------

    $ perl EvaluationTools/wa_check_align.pl alignments.out
    $ perl EvaluationTools/wa_eval_align.pl ../data/test.wa.nonullalign alignments.out
