# filename: feature_engineering.py
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the parquet files into pandas dataframes
labs = pd.read_parquet('data/filtered_patients_labs.parquet')
patient_info = pd.read_parquet('data/patients_info.parquet')
labels = pd.read_parquet('data/patients_labels.parquet')

# Generate a list of lab test columns, excluding 'hadm_id', 'charttime'
lab_tests = [col for col in labs.columns if col not in ['hadm_id', 'charttime']]

# Group by 'hadm_id' and 'charttime', then take the mean for each column
grouped_labs = labs.groupby(['hadm_id', 'charttime']).mean().reset_index()

# Sort the grouped_labs dataframe by 'hadm_id' and 'charttime'
grouped_labs.sort_values(by=['hadm_id', 'charttime'], inplace=True)

# Calculate new features for each lab test
for lab_test in lab_tests:
    # Baseline value (first value for each hadm_id)
    baseline = grouped_labs.groupby('hadm_id')[lab_test].transform('first')
    grouped_labs[f'{lab_test}_baseline_delta'] = grouped_labs[lab_test] - baseline
    
    # Delta from previous value
    grouped_labs[f'{lab_test}_diff'] = grouped_labs.groupby('hadm_id')[lab_test].diff()
    
    # Time difference in hours from previous value
    grouped_labs['charttime_diff'] = grouped_labs.groupby('hadm_id')['charttime'].diff().dt.total_seconds() / 3600
    grouped_labs[f'{lab_test}_rateofchange'] = grouped_labs[f'{lab_test}_diff'] / grouped_labs['charttime_diff']

# Drop unnecessary columns
grouped_labs.drop(columns=['charttime', 'charttime_diff'], inplace=True)

# Group by hadm_id and aggregate
aggregations = {col: ['mean', 'median', 'std', 'min', 'max'] for col in grouped_labs.columns if col != 'hadm_id'}
engineered_features = grouped_labs.groupby('hadm_id').agg(aggregations)

# Flatten the multi-index columns
engineered_features.columns = ['_'.join(col).strip() for col in engineered_features.columns.values]

# Fix the column names by removing trailing underscores
engineered_features.columns = [col.rstrip('_') for col in engineered_features.columns]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
engineered_features = pd.DataFrame(imputer.fit_transform(engineered_features), columns=engineered_features.columns, index=engineered_features.index)

# Merge patient_info with engineered_features
features = patient_info.merge(engineered_features, on='hadm_id', how='left')

# Merge labels with features
features_labels = features.merge(labels, on='hadm_id', how='left')

# Drop any rows with missing values
features_labels.dropna(inplace=True)

# Drop the 'hadm_id' column
features_labels.drop(columns=['hadm_id'], inplace=True)

# Save the final dataframe
features_labels.to_parquet('data/features_labels.parquet')