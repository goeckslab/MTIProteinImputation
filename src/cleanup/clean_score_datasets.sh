model=$1

cd src/cleanup || exit

python3 clean_score_datasets.py --model "${model}"
