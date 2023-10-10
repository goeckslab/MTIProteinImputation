

echo "Preparing data..."
./src/data_preparation/prepare_spatial_data.sh

echo "Data preparation complete."
echo "Running experiments..."
echo "Running EN experiments..."
./src/en/run_experiments.sh 60
echo "EN experiments done."

echo "Running LGBM experiments..."
./src/lgbm/run_experiments.sh
echo "LGBM experiments done."

echo "Running AE Single experiments..."
./src/ae/run_experiments.sh
echo "AE Single experiments done."

echo "Running AE Multi experiments..."
./src/ae_m/run_experiments.sh
echo "AE Multi experiments done."

echo "Creating figures, tables and supplementary material..."
./src/figures/create_figures.sh
echo "Figures, tables and supplementary material created."

echo "Script complete."