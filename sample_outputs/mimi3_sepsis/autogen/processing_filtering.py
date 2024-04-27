# filename: processing_filtering.py
import pandas as pd
import json

# Load 'data/patients_labs.parquet' into pandas dataframe (labs)
labs = pd.read_parquet('data/patients_labs.parquet')

# Load 'data/lab_test_types.json' and create a list of lab test types (lab_test_types)
with open('data/lab_test_types.json', 'r') as file:
    lab_test_types = json.load(file)

# Remove any values in the lab_test_types list that do not exist in the columns of labs dataframe
lab_test_types = [test for test in lab_test_types if test in labs.columns]

# Remove any columns (except hadm_id, charttime) in the labs dataframe that do not exist in the list of lab_test_types
required_columns = ['hadm_id', 'charttime'] + lab_test_types
labs = labs[required_columns]

# Remove any rows where all the lab_test_types columns are null
labs.dropna(subset=lab_test_types, how='all', inplace=True)

# Save the labs dataframe to 'data/filtered_patients_labs.parquet'
labs.to_parquet('data/filtered_patients_labs.parquet')