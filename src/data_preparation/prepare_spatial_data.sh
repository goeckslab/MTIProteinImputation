#!/bin/bash

bxs=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')
#radius=(23 46 92 138 184)
radius=( [23]=15 [46]=30 [92]=60 [138]=90 [184]=120 )

# iterate trough all biopsies and radius and generate engineered features
for bx in "${bxs[@]}"; do
  for r in "${!radius[@]}"; do
    microns="${radius[$r]}"
    px="${r}"

    echo "Processing biopsy ${bx} with radius ${microns} µm"

    python3 ./src/data_preparation/create_engineered_features.py --file data/bxs/"${bx}".csv -r "${px}" --output data/bxs_"${microns}"_µm/
  done
done



