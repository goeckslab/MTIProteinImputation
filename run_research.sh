

echo "Preparing data..."
./src/data_preparation/download_data.sh
./src/data_preparation/prepare_spatial_data.sh

echo "Data preparation complete."
echo "Running experiments..."

echo "Running null model experiments..."
python3 src/null_model/run_experiments.sh
echo "Null model experiments done."

echo "Running EN experiments..."
./src/en/run_experiments.sh 30
echo "EN experiments done."

echo "Running LGBM experiments..."
./src/lgbm/run_experiments.sh 30
echo "LGBM experiments done."

echo "Running AE Single experiments..."
./src/ae/run_experiments.sh 30
echo "AE Single experiments done."

echo "Running AE Multi experiments..."
./src/ae_m/run_experiments.sh 30
echo "AE Multi experiments done."

echo "Cleaning up..."
./src/cleanup/clean_score_datasets.sh
echo "Clean up complete."

echo "Running downstream tasks..."
./src/classifier/run_downstream_classification.sh
echo "Downstream tasks complete."

echo "Creating figures, tables and supplementary material..."
./src/figures/create_figures.sh
echo "Figures, tables and supplementary material created."

echo "Script complete."