#!/usr/bin/env bash

files="ibm-1.log ibm-2.log ibm-2-rand-1.log ibm-2-rand-2.log ibm-2-rand-3.log ibm-2-uniform.log"
declare -A temp_files

for file in $files;
do
  temp_files[$file]=$(mktemp);
done

for file in $files;
do
  cat $file | grep "AER" | tail -30 | awk '{ print $6 }' | sed 's/,//' > ${temp_files[$file]}
done

all_temp_files=$(for file in $files; do echo ${temp_files[$file]}; done | xargs)

paste $all_temp_files
