TEST_ID=$(test_id)
MARKER=$(marker)
PATIENT_ID=$(patient_id)
OUTPUT_DIRECTORY = $(TEST_ID)/$(TEST_ID)/$(MARKER)
DATA_DIRECTORY = ../../../data/bxs
DATA_DIRECTORY_COMBINED = ../../../data/bxs/combined
DATA_DIRECTORY_COMBINED_PREPROCESSED = ../../../data/bxs/combined/preprocessed
DATA_DIRECTORY_PREPROCESSED = ../../../data/bxs/preprocessed
PREPROCESSED_DATASET = $(TEST_ID)_preprocessed_dataset.tsv
COMBINED_DATASET = $(PATIENT_ID)_excluded_dataset.csv
COMBINED_PREPROCESSED_DATA = $(PATIENT_ID)_excluded_dataset.tsv

# Create preprocessed dataset.
prepare_data:
	mkdir -p "$(DATA_DIRECTORY_PREPROCESSED)" && \
	mkdir -p "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)" && \
	python ../../shared_scripts/prepare_data.py "$(DATA_DIRECTORY)/$(TEST_ID).csv" "$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)" && \
	python ../../shared_scripts/prepare_data.py "$(DATA_DIRECTORY_COMBINED)/$(COMBINED_DATASET)" "$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PREPROCESSED_DATA)"


# Create combined dataset
combine_data_sets:
	mkdir -p "$(DATA_DIRECTORY_COMBINED)" && python ../../shared_scripts/combine_datasets.py --dir "$(DATA_DIRECTORY)" --target "$(TEST_ID)" --output_dir "$(DATA_DIRECTORY_COMBINED)" -exp

# Run a ludwig experiment using the preprocessed dataset. Results are placed into the $(SAMPLE) directory.
run_en: combine_data_sets prepare_data
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && python ../../../elastic_net.py --marker "$(MARKER)" --train "../../../$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PREPROCESSED_DATA)" --test "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"

run-en-exa:
	mkdir -p $(OUTPUT_DIRECTORY) && pushd $(OUTPUT_DIRECTORY) && python ../../../elastic_net.py --marker "$(MARKER)" --train "../../../$(DATA_DIRECTORY_COMBINED_PREPROCESSED)/$(COMBINED_PREPROCESSED_DATA)" --test "../../../$(DATA_DIRECTORY_PREPROCESSED)/$(PREPROCESSED_DATASET)"
