biopsy=$1
mode=$2
spatial=$3

echo "Biopsy: ${biopsy}" "Mode: ${mode}" "Spatial: ${spatial}"

if [ "$spatial" != "" ]; then
  python3 evaluate_lgbm_models.py -b $1 --mode $2 --spatial $3 -s 1
else
  python3 evaluate_lgbm_models.py -b $1 --mode $2 -s 1
fi
