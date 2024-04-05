# single

python3 src/classifier/treatment_classifier.py -m exp -p 9_2
python3 src/classifier/treatment_classifier.py -m exp -p 9_3
python3 src/classifier/treatment_classifier.py -m exp -p 9_14
python3 src/classifier/treatment_classifier.py -m exp -p 9_15


# multi
python3 src/classifier/treatment_classifier_multi.py -m exp -p 9_2
python3 src/classifier/treatment_classifier_multi.py -m exp -p 9_3
python3 src/classifier/treatment_classifier_multi.py -m exp -p 9_14
python3 src/classifier/treatment_classifier_multi.py -m exp -p 9_15


python3 src/classifier/pycaret.py -m exp -p 9_2