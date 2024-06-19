#!/bin/bash

repetitions=$1

if [ "$repetitions" == "" ]; then
  repetitions=1
fi

echo "Iterations:" ${repetitions}

markers=('AR' 'Ki67' 'CK14' 'aSMA' 'ER' 'HER2' 'EGFR' 'p21' 'Vimentin' 'Ecad' 'CK17' 'pERK' 'PR' 'pRB' 'CK19')
cores=('A8' 'H8' 'G9' 'E5' 'C1' 'D2' 'G3' 'B9' 'A5' 'B10' 'H7' 'H3' 'G7' 'F2' 'H2' 'E8' 'A10' 'B3' 'H6' 'E10' 'G1' 'E6' 'C10' 'B7' 'D8' 'E9' 'F7' 'E3' 'F1' 'B6' 'B1' 'A6' 'H10' 'H4')

batch_size=5
count=0

for i in $(seq 1 $repetitions); do
  for marker in "${markers[@]}"; do
    for core in "${cores[@]}"; do
      echo "Running marker=${marker} core=${core}"
      python3 src/tma/lgbm.py --marker "${marker}" --core "${core}" &
      ((count++))

      if [ "$count" -ge "$batch_size" ]; then
        wait # wait for the current batch to finish
        echo "Starting a new batch..."
        count=0 # reset the counter
      fi
    done
  done
done

# wait for any remaining background processes
wait

echo "All scripts have finished executing."
