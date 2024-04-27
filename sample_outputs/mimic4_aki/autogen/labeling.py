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
# Note: Assuming 'labs dataframe' mentioned for hadm_id uniqueness is a typo and meant 'diagnoses dataframe'.
# If labs dataframe is different and needs to be loaded, this script needs to be adjusted accordingly.
labels = pd.DataFrame({'hadm_id': diagnoses['hadm_id'].unique()})
labels['condition_label'] = labels['hadm_id'].apply(lambda x: 1 if x in positive_diagnoses else 0)

# Save the labels dataframe as "data/patients_labels.parquet"
labels.to_parquet('data/patients_labels.parquet')