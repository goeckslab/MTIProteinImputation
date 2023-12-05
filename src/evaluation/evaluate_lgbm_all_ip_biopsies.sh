# Always add a trailing slash to the folder paths

biopsies=('9_2_1' '9_2_2' '9_3_1' '9_3_2' '9_14_1' '9_14_2' '9_15_1' '9_15_2')

cd src/evaluation || exit

for biopsy in "${biopsies[@]}"; do
   ./evaluate_lgbm_models.sh "${biopsy}" "ip" 0
   ./evaluate_lgbm_models.sh "${biopsy}" "ip" 15
   ./evaluate_lgbm_models.sh "${biopsy}" "ip" 30
   ./evaluate_lgbm_models.sh "${biopsy}" "ip" 60
   ./evaluate_lgbm_models.sh "${biopsy}" "ip" 90
   ./evaluate_lgbm_models.sh "${biopsy}" "ip" 120
done