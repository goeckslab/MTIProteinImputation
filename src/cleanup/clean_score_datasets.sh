cd src/cleanup || exit

python3 clean_score_datasets.py --model lgbm
python3 clean_score_datasets.py --model ae
python3 clean_score_datasets.py --model ae_m
