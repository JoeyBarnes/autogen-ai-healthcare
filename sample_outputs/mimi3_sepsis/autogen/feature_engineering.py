# filename: feature_engineering.py
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the parquet files into pandas dataframes
labs = pd.read_parquet('data/filtered_patients_labs.parquet')
patient_info = pd.read_parquet('data/patients_info.parquet')
labels = pd.read_parquet('data/patients_labels.parquet')

# Generate a list of lab test columns excluding 'hadm_id', 'charttime'
lab_tests = [col for col in labs.columns if col not in ['hadm_id', 'charttime']]

# Group labs dataframe by 'hadm_id' and 'charttime' and take the mean for each column
grouped_labs = labs.groupby(['hadm_id', 'charttime']).mean().reset_index()

# Sort the grouped_labs dataframe by 'hadm_id' and 'charttime'
grouped_labs.sort_values(by=['hadm_id', 'charttime'], inplace=True)

# Calculate the engineered features for each lab_test
for lab_test in lab_tests:
    grouped_labs[f'{lab_test}_baseline_delta'] = grouped_labs.groupby('hadm_id')[lab_test].transform(lambda x: x - x.iloc[0])
    grouped_labs[f'{lab_test}_diff'] = grouped_labs.groupby('hadm_id')[lab_test].diff()
    grouped_labs[f'{lab_test}_timediff'] = grouped_labs.groupby('hadm_id')['charttime'].diff().dt.total_seconds() / 3600
    grouped_labs[f'{lab_test}_rateofchange'] = grouped_labs[f'{lab_test}_diff'] / grouped_labs[f'{lab_test}_timediff']

# Drop the specified columns
grouped_labs.drop(['charttime', f'{lab_test}_timediff'], axis=1, inplace=True)

# Group by 'hadm_id' and aggregate
engineered_features = grouped_labs.groupby('hadm_id').agg(['mean', 'median', 'std', 'min', 'max'])

# Flatten the multi-index columns
engineered_features.columns = ['_'.join(col).strip() for col in engineered_features.columns.values]

# Fix the column names by removing trailing underscores
engineered_features.columns = [col.rstrip('_') for col in engineered_features.columns]

# Impute engineered_features to fill any missing values
imputer = SimpleImputer(strategy='mean')
engineered_features_imputed = pd.DataFrame(imputer.fit_transform(engineered_features), columns=engineered_features.columns, index=engineered_features.index)

# Merge the dataframes
features = pd.merge(patient_info, engineered_features_imputed, on='hadm_id', how='inner')
features_labels = pd.merge(features, labels, on='hadm_id', how='inner')

# Drop any rows with missing values
features_labels.dropna(inplace=True)

# Drop the 'hadm_id' column
features_labels.drop('hadm_id', axis=1, inplace=True)

# Save the final dataframe
features_labels.to_parquet('data/features_labels.parquet')