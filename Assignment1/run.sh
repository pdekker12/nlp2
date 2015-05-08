#!/usr/bin/env bash

nohup sh run-ibm-1.sh > ibm-1.log &
nohup sh run-ibm-2.sh > ibm-2.log &
nohup sh run-ibm-2-rand.sh > ibm-2-rand-1.log &
nohup sh run-ibm-2-rand.sh > ibm-2-rand-2.log &
nohup sh run-ibm-2-rand.sh > ibm-2-rand-3.log &
nohup sh run-ibm-2-uniform.sh > ibm-2-uniform.log &
