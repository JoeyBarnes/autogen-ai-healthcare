# filename: dimensionality_reduction.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the parquet file into a pandas dataframe
features_labels = pd.read_parquet('data/features_labels.parquet')

# Split the dataframe into features and labels
labels = features_labels['condition_label']
features = features_labels.drop('condition_label', axis=1)

# Standardize the features
features_standardized = StandardScaler().fit_transform(features)

# Perform dimensionality reduction using PCA, retaining enough components to explain 95% of the variance
pca = PCA(n_components=0.95)
reduced_features = pca.fit_transform(features_standardized)

# Convert column names of the reduced features to strings
reduced_features_df = pd.DataFrame(reduced_features, columns=[str(i) for i in range(reduced_features.shape[1])])

# Combine the reduced features and labels into a single dataframe
reduced_features_labels = pd.concat([reduced_features_df, labels.reset_index(drop=True)], axis=1)

# Save the combined dataframe to a new parquet file
reduced_features_labels.to_parquet('data/reduced_features_labels.parquet')

# Print the original and reduced number of features
print(f"Original number of features: {features.shape[1]}")
print(f"Number of features after reduction: {reduced_features_df.shape[1]}")