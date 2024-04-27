# filename: processing_filtering.py
import pandas as pd
import json

# Load the labs dataframe from a parquet file
labs = pd.read_parquet('data/patients_labs.parquet')

# Load the lab test types from a JSON file
with open('data/lab_test_types.json', 'r') as file:
    lab_test_types = json.load(file)

# Remove any values from lab_test_types that do not exist in the labs dataframe columns
lab_test_types = [test for test in lab_test_types if test in labs.columns]

# Remove any columns in the labs dataframe that do not exist in lab_test_types
# while keeping 'hadm_id' and 'charttime'
columns_to_keep = ['hadm_id', 'charttime'] + lab_test_types
labs = labs[columns_to_keep]

# Remove any rows where all the lab_test_types columns are null
labs.dropna(subset=lab_test_types, how='all', inplace=True)

# Save the filtered labs dataframe to a new parquet file
labs.to_parquet('data/filtered_patients_labs.parquet')