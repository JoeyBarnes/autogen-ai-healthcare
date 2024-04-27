# filename: processing_filtering.py
import pandas as pd
import json

# Load the labs dataframe
labs = pd.read_parquet('data/patients_labs.parquet')

# Load the lab test types
with open('data/lab_test_types.json', 'r') as file:
    lab_test_types = json.load(file)

# Remove any values in lab_test_types list that do not exist in the columns of labs dataframe
lab_test_types = [test for test in lab_test_types if test in labs.columns]

# Remove any columns in labs dataframe that do not exist in the list of lab_test_types, except 'hadm_id', 'charttime'
labs = labs[['hadm_id', 'charttime'] + lab_test_types]

# Remove any rows where all the lab_test_types columns are null
labs.dropna(subset=lab_test_types, how='all', inplace=True)

# Save the filtered labs dataframe
labs.to_parquet('data/filtered_patients_labs.parquet')