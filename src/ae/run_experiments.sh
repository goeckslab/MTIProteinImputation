iterations=$1

if [ "$iterations" == "" ]; then
  iterations=1
fi


# multi imputation
echo "Multi imputation"
./src/ae/multi_imputation/schedule_experiments.sh ip mean 1 0 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip mean 1 15 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip mean 1 30 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip mean 1 60 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip mean 1 90 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip mean 1 120 $iterations

./src/ae/multi_imputation/schedule_experiments.sh exp mean 1 0 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp mean 1 15 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp mean 1 30 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp mean 1 60 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp mean 1 90 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp mean 1 120 $iterations


./src/ae/multi_imputation/schedule_experiments.sh ip zero 1 0 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip zero 1 15 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip zero 1 30 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip zero 1 60 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip zero 1 90 $iterations
./src/ae/multi_imputation/schedule_experiments.sh ip zero 1 120 $iterations

./src/ae/multi_imputation/schedule_experiments.sh exp zero 1 0 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp zero 1 15 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp zero 1 30 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp zero 1 60 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp zero 1 90 $iterations
./src/ae/multi_imputation/schedule_experiments.sh exp zero 1 120 $iterations

# combine scores
echo "Combining multi imputation scores"
python3 src/cleanup/combine_ae_scores.py --model ae_m
# clean scores
echo "Cleaning multi imputation scores"
./src/cleanup/clean_score_datasets.sh ae_m

# single imputation
echo "Single imputation"
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 0 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 15 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 30 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 60 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 90 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 120 $iterations

./src/ae/single_imputation/schedule_experiments.sh exp mean 1 0 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp mean 1 15 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp mean 1 30 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp mean 1 60 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp mean 1 90 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp mean 1 120 $iterations


./src/ae/single_imputation/schedule_experiments.sh ip zero 1 0 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip zero 1 15 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip zero 1 30 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip zero 1 60 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip zero 1 90 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip zero 1 120 $iterations

./src/ae/single_imputation/schedule_experiments.sh exp zero 1 0 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp zero 1 15 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp zero 1 30 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp zero 1 60 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp zero 1 90 $iterations
./src/ae/single_imputation/schedule_experiments.sh exp zero 1 120 $iterations


# combine scores
echo "Combining single imputation scores"
python3 src/cleanup/combine_ae_scores.py --model ae
# clean scores
echo "Cleaning single imputation scores"
./src/cleanup/clean_score_datasets.sh ae