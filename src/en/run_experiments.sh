iterations=$1

if [ "$iterations" == "" ]; then
  iterations=1
fi

cd ./src/en/in_patient || exit
#./evaluate_all_marker_in_patient.sh 9_2_1 9_2_2 $iterations
#./evaluate_all_marker_in_patient.sh 9_2_2 9_2_1 $iterations
#./evaluate_all_marker_in_patient.sh 9_3_1 9_3_2 $iterations
#./evaluate_all_marker_in_patient.sh 9_3_2 9_3_1 $iterations
#./evaluate_all_marker_in_patient.sh 9_14_1 9_14_2 $iterations
#./evaluate_all_marker_in_patient.sh 9_14_2 9_14_1 $iterations
#./evaluate_all_marker_in_patient.sh 9_15_1 9_15_2 $iterations
#./evaluate_all_marker_in_patient.sh 9_15_2 9_15_1 $iterations

cd ../../../ || exit
cd ./src/en/exp_patient || exit
./evaluate_all_marker_exp_patient.sh 9_2_1 9_2 $iterations
./evaluate_all_marker_exp_patient.sh 9_2_2 9_2 $iterations
./evaluate_all_marker_exp_patient.sh 9_3_1 9_3 $iterations
./evaluate_all_marker_exp_patient.sh 9_3_2 9_3 $iterations
./evaluate_all_marker_exp_patient.sh 9_14_1 9_14 $iterations
./evaluate_all_marker_exp_patient.sh 9_14_2 9_14 $iterations
./evaluate_all_marker_exp_patient.sh 9_15_1 9_15 $iterations
./evaluate_all_marker_exp_patient.sh 9_15_2 9_15 $iterations