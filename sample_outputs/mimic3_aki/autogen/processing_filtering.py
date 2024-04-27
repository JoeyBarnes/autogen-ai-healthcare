# filename: processing_filtering.py
import pandas as pd
import json

# Load the labs dataframe from a parquet file
labs = pd.read_parquet('data/patients_labs.parquet')

# Load the lab test types from a JSON file
with open('data/lab_test_types.json', 'r') as f:
    lab_test_types = json.load(f)

# Remove any values in the lab_test_types list that do not exist in the columns of labs dataframe
lab_test_types = [test for test in lab_test_types if test in labs.columns]

# Remove any columns in the labs dataframe that do not exist in the list of lab_test_types
# while keeping 'hadm_id' and 'charttime'
columns_to_keep = ['hadm_id', 'charttime'] + lab_test_types
labs = labs[columns_to_keep]

# Remove any rows where all the lab_test_types columns are null
labs = labs.dropna(subset=lab_test_types, how='all')

# Save the labs dataframe to a new parquet file
labs.to_parquet('data/filtered_patients_labs.parquet')