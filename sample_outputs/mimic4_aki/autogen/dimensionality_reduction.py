# filename: dimensionality_reduction.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the parquet file into a pandas dataframe
features_labels = pd.read_parquet('data/features_labels.parquet')

# Split the dataframe into features and labels
features = features_labels.drop('condition_label', axis=1)
labels = features_labels[['condition_label']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform dimensionality reduction using PCA
pca = PCA(n_components=0.95)  # retain 95% of the variance
features_reduced = pca.fit_transform(features_scaled)

# Convert the column names of the reduced features to strings
features_reduced_df = pd.DataFrame(features_reduced, columns=[str(i) for i in range(features_reduced.shape[1])])

# Combine the reduced features and labels
reduced_features_labels = pd.concat([features_reduced_df, labels.reset_index(drop=True)], axis=1)

# Save the combined dataframe to a new parquet file
reduced_features_labels.to_parquet('data/reduced_features_labels.parquet')

# Print the original and reduced number of features
print(f"Original number of features: {features.shape[1]}")
print(f"Number of features after reduction: {features_reduced.shape[1]}")