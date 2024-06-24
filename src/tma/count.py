from patient_mapping import patient_mapping
import os
import pandas as pd
from pathlib import Path

# Initialize a dictionary to count occurrences of each patient
patient_counts = {}

# Iterate over the position mapping dictionary and count occurrences of each patient
for patient in patient_mapping.values():
    if patient in patient_counts:
        patient_counts[patient] += 1
    else:
        patient_counts[patient] = 1

# Print the counts
for patient, count in patient_counts.items():
    print(f'{patient}: {count}')

patients = []
# iterate thorugh data/tma/base
for root, dirs, files in os.walk("data/tma/base"):
    for core in files:
        # get patient mapping
        patient = patient_mapping[Path(core).stem]
        patients.append({
            "Patient": patient,
        })

patients_df = pd.DataFrame(patients)
# print unique patients
print(len(patients_df["Patient"].unique()))

