from patient_mapping import patient_mapping


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
