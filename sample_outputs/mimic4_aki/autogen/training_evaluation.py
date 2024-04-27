# filename: training_evaluation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
import pickle
import json
import os

# Load the data
df = pd.read_parquet('data/reduced_features_labels.parquet')

# Split the dataframe into features (X) and labels (y)
X = df.drop('condition_label', axis=1)
y = df['condition_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
classifiers = {
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2,
                                                     min_samples_leaf=2, random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=300, random_state=42),
    "MLPClassifier": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

model_details = {}

for model_name, model in classifiers.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_dict = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    pr_dict = {"precision": precision.tolist(), "recall": recall.tolist(), "auc": pr_auc}
    
    # Save the model
    model_path = f'data/{model_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save details
    model_details[model_name] = {
        "classification_report": report,
        "roc_curve": roc_dict,
        "pr_curve": pr_dict,
        "model_path": model_path
    }

# Save model details as JSON
with open('data/model_details.json', 'w') as f:
    json.dump(model_details, f, indent=4)

# Load and print classification reports
for model_name, details in model_details.items():
    report_df = pd.DataFrame(details['classification_report']).transpose()
    print(f"## {model_name} Classification Report\n")
    print(report_df.to_markdown())