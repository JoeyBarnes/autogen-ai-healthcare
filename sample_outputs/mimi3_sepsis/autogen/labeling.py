# filename: labeling.py
import pandas as pd

# Load the parquet files into pandas dataframes
diagnoses = pd.read_parquet('data/patients_diagnoses.parquet')
icd_codes = pd.read_parquet('data/icd_codes.parquet')

# Create a list of icd_codes (condition_codes) where the long_title column contains
# any of the following keywords: ['sepsis', 'septic shock', 'bacteremia', 'septicemia'], case insensitive
keywords = ['sepsis', 'septic shock', 'bacteremia', 'septicemia']
condition_codes = icd_codes[icd_codes['long_title'].str.contains('|'.join(keywords), case=False)]['icd_code'].tolist()

# Create a unique list of hadm_ids (positive_diagnoses) from diagnoses dataframe
# where the icd_code is in the condition_codes list
positive_diagnoses = diagnoses[diagnoses['icd_code'].isin(condition_codes)]['hadm_id'].unique().tolist()

# Create a new dataframe (labels) with the following columns:
# - hadm_id (unique from labs dataframe)
# - condition_label (1 if hadm_id is in positive_diagnoses list, 0 otherwise)
# Note: Assuming 'labs dataframe' refers to 'diagnoses dataframe' for hadm_id uniqueness
labels = pd.DataFrame({'hadm_id': diagnoses['hadm_id'].unique()})
labels['condition_label'] = labels['hadm_id'].apply(lambda x: 1 if x in positive_diagnoses else 0)

# Save the labels as "data/patients_labels.parquet"
labels.to_parquet('data/patients_labels.parquet')