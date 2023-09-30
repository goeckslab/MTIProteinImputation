iterations=$1

if [ "$iterations" == "" ]; then
  iterations=1
fi


./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_2_1 9_2_2 $iterations
./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_2_2 9_2_1 $iterations
./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_3_1 9_3_2 $iterations
./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_3_2 9_3_1 $iterations
./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_14_1 9_14_2 $iterations
./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_14_2 9_14_1 $iterations
./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_15_1 9_15_2 $iterations
./src/en/in_patient/evaluate_all_marker_in_patient.sh 9_15_2 9_15_1 $iterations

./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_2_1 $iterations
./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_2_2 $iterations
./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_3_1 $iterations
./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_3_2 $iterations
./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_14_1 $iterations
./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_14_2 $iterations
./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_15_1 $iterations
./src/en/exp_patient/evaluate_all_marker_exp_patient.sh 9_15_2 $iterations