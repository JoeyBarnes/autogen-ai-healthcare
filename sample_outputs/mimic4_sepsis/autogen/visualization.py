# filename: visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load model performance details
with open('data/model_details.json') as file:
    model_details = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(model_details).T

# Classification Report Data
cr_data = {}
for model, details in model_details.items():
    cr = details['classification_report']['weighted avg']
    accuracy = details['classification_report']['accuracy']
    cr_data[model.replace("Classifier", "")] = {
        'Precision': cr['precision'],
        'Recall': cr['recall'],
        'F1-Score': cr['f1-score'],
        'Accuracy': accuracy
    }

# Plotting Classification Report
cr_df = pd.DataFrame(cr_data).T.melt(var_name='Performance Metric', value_name='Value', ignore_index=False).reset_index().rename(columns={'index': 'Model'})
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Value', hue='Performance Metric', data=cr_df)
plt.ylim(min(cr_df['Value']) * 0.95, max(cr_df['Value']) * 1.05)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/classification_report.png')

# ROC Curve Data
roc_curve_data = {}
for model, details in model_details.items():
    roc = details['roc_curve']
    roc_curve_data[model.replace("Classifier", "")] = roc

# Plotting ROC Curve
plt.figure(figsize=(10, 6))
for model, roc in roc_curve_data.items():
    plt.plot(roc['fpr'], roc['tpr'], label=f"{model} (AUC = {roc['auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('data/roc_curve.png')

# Precision-Recall Curve Data
pr_curve_data = {}
for model, details in model_details.items():
    pr = details['pr_curve']
    pr_curve_data[model.replace("Classifier", "")] = pr

# Plotting Precision-Recall Curve
plt.figure(figsize=(10, 6))
for model, pr in pr_curve_data.items():
    plt.plot(pr['recall'], pr['precision'], label=f"{model} (AUC = {pr['auc']:.2f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig('data/pr_curve.png')

# Output links to the plots
print("Links to the plots:")
print("Classification Report: 'data/classification_report.png'")
print("ROC Curve: 'data/roc_curve.png'")
print("Precision-Recall Curve: 'data/pr_curve.png'")