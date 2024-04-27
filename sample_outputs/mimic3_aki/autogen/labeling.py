# filename: labeling.py
import pandas as pd

# Load the parquet files into pandas dataframes
diagnoses = pd.read_parquet('data/patients_diagnoses.parquet')
icd_codes = pd.read_parquet('data/icd_codes.parquet')

# Create a list of icd_codes (condition_codes) where the long_title column contains any of the keywords
keywords = ['acute kidney injury', 'acute kidney failure', 'aki']
condition_codes = icd_codes[icd_codes['long_title'].str.contains('|'.join(keywords), case=False)]['icd_code'].tolist()

# Create a unique list of hadm_ids (positive_diagnoses) from diagnoses dataframe where the icd_code is in the condition_codes list
positive_diagnoses = diagnoses[diagnoses['icd_code'].isin(condition_codes)]['hadm_id'].unique().tolist()

# Create a new dataframe (labels) with the specified columns
# First, create a unique list of hadm_id from diagnoses dataframe
unique_hadm_ids = diagnoses['hadm_id'].unique()

# Then, create the condition_label column
condition_label = [1 if hadm_id in positive_diagnoses else 0 for hadm_id in unique_hadm_ids]

# Finally, create the labels dataframe
labels = pd.DataFrame({'hadm_id': unique_hadm_ids, 'condition_label': condition_label})

# Save the labels dataframe as a parquet file
labels.to_parquet('data/patients_labels.parquet')

print("Labels dataframe created and saved successfully.")