#!/bin/bash

mode=$1
replace_value=$2
experiments=$3
spatial=$4

biopsies=('G5' 'A1' 'E6' 'G3' 'B6' 'F1' 'C10' 'A8' 'F9' 'H4' 'B3' 'G6' 'G1' 'E5' 'G7' 'A4' 'E4'])

# iterate through all biopsies
for biopsy in "${biopsies[@]}"; do
  for i in $(seq 1 $experiments)
  do
      echo biopsy="${biopsy}" mode="${mode}" replace_value="${replace_value}" spatial="${spatial}" biopsy="${biopsy}"
      ./src/ae/single_imputation/ae_experiment.sh "${biopsy}" "${mode}" "${replace_value}" "${spatial}"
  done
done