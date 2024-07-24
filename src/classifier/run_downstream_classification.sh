
# create informative tiles
python3 src/classifier/informative_tile_creation.py -p 9_2
python3 src/classifier/informative_tile_creation.py -p 9_3
python3 src/classifier/informative_tile_creation.py -p 9_14
python3 src/classifier/informative_tile_creation.py -p 9_15

python3 src/post_processing/find_predictive_tissue.py

python3 src/classifier/single_cell_classifier.py -p 9_2
python3 src/classifier/single_cell_classifier.py -p 9_3
python3 src/classifier/single_cell_classifier.py -p 9_14
python3 src/classifier/single_cell_classifier.py -p 9_15