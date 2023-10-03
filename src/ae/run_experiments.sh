iterations=$1

if [ "$iterations" == "" ]; then
  iterations=1
fi

./src/ae/single_imputation/schedule_experiments.sh ip mean 1 0 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 15 $iterations
./src/ae/single_imputation/schedule_experiments.sh ip mean 1 30 $iterations^
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


# mutli imputation
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