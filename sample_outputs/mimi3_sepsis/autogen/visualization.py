# filename: visualization.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load model performance details
with open('data/model_details.json') as f:
    model_details = json.load(f)

# Prepare classification report data
cr_data = {model: {metric: details['classification_report']['weighted avg'][metric]
                   for metric in ['precision', 'recall', 'f1-score']}
           for model, details in model_details.items()}
for model in cr_data:
    cr_data[model]['accuracy'] = model_details[model]['classification_report']['accuracy']

# Convert to DataFrame for plotting
cr_df = pd.DataFrame(cr_data).T.reset_index()
cr_df = cr_df.melt(id_vars="index", var_name="Performance Metric", value_name="Value")
cr_df['index'] = cr_df['index'].str.replace('Classifier', '')

# Plot classification report data
plt.figure(figsize=(10, 6))
sns.barplot(x='index', y='Value', hue='Performance Metric', data=cr_df)
plt.ylim(min(cr_df['Value']) * 0.95, max(cr_df['Value']) * 1.05)
plt.xticks(rotation=45)
plt.title('Model Performance Comparison')
plt.savefig('data/classification_report.png')
plt.close()

# Prepare ROC curve data
roc_curve_data = {model: details['roc_curve'] for model, details in model_details.items()}

# Plot ROC curve
plt.figure(figsize=(10, 6))
for model, data in roc_curve_data.items():
    plt.plot(data['fpr'], data['tpr'], label=f"{model.replace('Classifier', '')} (AUC = {data['auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('data/roc_curve.png')
plt.close()

# Prepare precision-recall data
pr_curve_data = {model: details['pr_curve'] for model, details in model_details.items()}

# Plot Precision-Recall curve
plt.figure(figsize=(10, 6))
for model, data in pr_curve_data.items():
    plt.plot(data['recall'], data['precision'], label=f"{model.replace('Classifier', '')} (AUC = {data['auc']:.2f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.savefig('data/pr_curve.png')
plt.close()

# Output links to the plots
print("Links to the plots:")
print("Classification Report: data/classification_report.png")
print("ROC Curve: data/roc_curve.png")
print("Precision-Recall Curve: data/pr_curve.png")