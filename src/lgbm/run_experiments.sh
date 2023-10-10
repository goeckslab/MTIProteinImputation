iterations=$1

if [ "$iterations" == "" ]; then
  iterations=1
fi

cd ./src/lgbm/exp_patient || exit
for i in $(seq 1 $iterations); do
  cd ../exp_patient || exit
  echo "Creating exp_patient"
  ./evaluate_all_marker_exp_patient.sh 9_2_1 9_2
  ./evaluate_all_marker_exp_patient.sh 9_2_2 9_2
  ./evaluate_all_marker_exp_patient.sh 9_3_1 9_3
  ./evaluate_all_marker_exp_patient.sh 9_3_2 9_3
  ./evaluate_all_marker_exp_patient.sh 9_14_1 9_14
  ./evaluate_all_marker_exp_patient.sh 9_14_2 9_14
  ./evaluate_all_marker_exp_patient.sh 9_15_1 9_15
  ./evaluate_all_marker_exp_patient.sh 9_15_2 9_15

  cd ../exp_patient_15_µm || exit
  echo "Creating exp_patient_15_µm"
  ./evaluate_all_marker_exp_patient.sh 9_2_1 9_2
  ./evaluate_all_marker_exp_patient.sh 9_2_2 9_2
  ./evaluate_all_marker_exp_patient.sh 9_3_1 9_3
  ./evaluate_all_marker_exp_patient.sh 9_3_2 9_3
  ./evaluate_all_marker_exp_patient.sh 9_14_1 9_14
  ./evaluate_all_marker_exp_patient.sh 9_14_2 9_14
  ./evaluate_all_marker_exp_patient.sh 9_15_1 9_15
  ./evaluate_all_marker_exp_patient.sh 9_15_2 9_15

  cd ../exp_patient_30_µm || exit
  echo "Creating exp_patient_30_µm"
  ./evaluate_all_marker_exp_patient.sh 9_2_1 9_2
  ./evaluate_all_marker_exp_patient.sh 9_2_2 9_2
  ./evaluate_all_marker_exp_patient.sh 9_3_1 9_3
  ./evaluate_all_marker_exp_patient.sh 9_3_2 9_3
  ./evaluate_all_marker_exp_patient.sh 9_14_1 9_14
  ./evaluate_all_marker_exp_patient.sh 9_14_2 9_14
  ./evaluate_all_marker_exp_patient.sh 9_15_1 9_15
  ./evaluate_all_marker_exp_patient.sh 9_15_2 9_15

  cd ../exp_patient_60_µm || exit
  echo "Creating exp_patient_60_µm"
  ./evaluate_all_marker_exp_patient.sh 9_2_1 9_2
  ./evaluate_all_marker_exp_patient.sh 9_2_2 9_2
  ./evaluate_all_marker_exp_patient.sh 9_3_1 9_3
  ./evaluate_all_marker_exp_patient.sh 9_3_2 9_3
  ./evaluate_all_marker_exp_patient.sh 9_14_1 9_14
  ./evaluate_all_marker_exp_patient.sh 9_14_2 9_14
  ./evaluate_all_marker_exp_patient.sh 9_15_1 9_15
  ./evaluate_all_marker_exp_patient.sh 9_15_2 9_15

  cd ../exp_patient_90_µm || exit
  echo "Creating exp_patient_90_µm"
  ./evaluate_all_marker_exp_patient.sh 9_2_1 9_2
  ./evaluate_all_marker_exp_patient.sh 9_2_2 9_2
  ./evaluate_all_marker_exp_patient.sh 9_3_1 9_3
  ./evaluate_all_marker_exp_patient.sh 9_3_2 9_3
  ./evaluate_all_marker_exp_patient.sh 9_14_1 9_14
  ./evaluate_all_marker_exp_patient.sh 9_14_2 9_14
  ./evaluate_all_marker_exp_patient.sh 9_15_1 9_15
  ./evaluate_all_marker_exp_patient.sh 9_15_2 9_15

  cd ../exp_patient_120_µm || exit
  echo "Creating exp_patient_120_µm"
  ./evaluate_all_marker_exp_patient.sh 9_2_1 9_2
  ./evaluate_all_marker_exp_patient.sh 9_2_2 9_2
  ./evaluate_all_marker_exp_patient.sh 9_3_1 9_3
  ./evaluate_all_marker_exp_patient.sh 9_3_2 9_3
  ./evaluate_all_marker_exp_patient.sh 9_14_1 9_14
  ./evaluate_all_marker_exp_patient.sh 9_14_2 9_14
  ./evaluate_all_marker_exp_patient.sh 9_15_1 9_15
  ./evaluate_all_marker_exp_patient.sh 9_15_2 9_15

  cd ../in_patient || exit
  echo "Creating in_patient"
  ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
  ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
  ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
  ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
  ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
  ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
  ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
  ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1

  cd ../in_patient_15_µm || exit
  echo "Creating in_patient_15_µm"
  ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
  ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
  ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
  ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
  ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
  ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
  ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
  ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1

  cd ../in_patient_30_µm || exit
  echo "Creating in_patient_30_µm"
  ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
  ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
  ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
  ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
  ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
  ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
  ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
  ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1

  cd ../in_patient_60_µm || exit
  echo "Creating in_patient_60_µm"
  ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
  ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
  ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
  ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
  ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
  ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
  ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
  ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1


  cd ../in_patient_90_µm || exit
  echo "Creating in_patient_90_µm"
  ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
  ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
  ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
  ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
  ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
  ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
  ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
  ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1

  cd ../in_patient_120_µm || exit
  echo "Creating in_patient_120_µm"
  ./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2
  ./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1
  ./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2
  ./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1
  ./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2
  ./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1
  ./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2
  ./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1

done

# return to root dir
cd ../../.. || exit
echo "Evaluating all biopsies"
./src/evaluation/evaluate_lgbm_all_exp_biopsies.sh
./src/evaluation/evaluate_lgbm_all_in_biopsies.sh

echon "Cleaning score directories"
./src/cleanup/clean_score_datasets.sh lgbm


echo "Done"




