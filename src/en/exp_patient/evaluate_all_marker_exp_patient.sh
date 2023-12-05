test_id=$1
patient=$2
repetitions=$3

if [ "$repetitions" == "" ]; then
  repetitions=1
fi

echo "Iterations:" ${repetitions}

markers=('pRB' 'CD45' 'CK19' 'Ki67' 'aSMA' 'Ecad' 'PR' 'CK14' 'HER2' 'AR' 'CK17' 'p21' 'Vimentin' 'pERK' 'EGFR' 'ER')
for i in $(seq 1 $repetitions); do
  for marker in "${markers[@]}"; do
    echo test_id="${test_id}" marker="${marker}" patient_id="${patient}"
    make -f makefile run_en test_id="${test_id}" marker="${marker}" patient_id="${patient}"
  done
done
