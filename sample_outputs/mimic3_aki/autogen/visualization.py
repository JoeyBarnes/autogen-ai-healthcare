# filename: visualization.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model performance details into a pandas dataframe
with open('data/model_details.json') as file:
    model_details = json.load(file)

# Convert the JSON data into a structured DataFrame
df = pd.DataFrame.from_dict(model_details, orient='index')

# Prepare classification report data
cr_data = {}
for model, details in model_details.items():
    cr = details['classification_report']['weighted avg']
    accuracy = details['classification_report']['accuracy']
    cr_data[model.replace('Classifier', '')] = {
        'Precision': cr['precision'],
        'Recall': cr['recall'],
        'F1-Score': cr['f1-score'],
        'Accuracy': accuracy
    }

# Plotting the combined bar plot for classification report data
cr_df = pd.DataFrame.from_dict(cr_data, orient='index').reset_index().melt(id_vars='index')
cr_df.columns = ['Model', 'Performance Metric', 'Value']

plt.figure(figsize=(10, 6))
sns.barplot(data=cr_df, x='Model', y='Value', hue='Performance Metric')
plt.ylim(min(cr_df['Value']) * 0.95, max(cr_df['Value']) * 1.05)
plt.xticks(rotation=45)
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('data/classification_report.png')
plt.close()

# Prepare ROC curve data
roc_curve_data = {model.replace('Classifier', ''): details['roc_curve'] for model, details in model_details.items()}

# Plotting the combined line plot for ROC curve
plt.figure(figsize=(10, 6))
for model, roc in roc_curve_data.items():
    plt.plot(roc['fpr'], roc['tpr'], label=f"{model} (AUC = {roc['auc']:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig('data/roc_curve.png')
plt.close()

# Prepare Precision-Recall curve data
pr_curve_data = {model.replace('Classifier', ''): details['pr_curve'] for model, details in model_details.items()}

# Plotting the combined line plot for Precision-Recall curve
plt.figure(figsize=(10, 6))
for model, pr in pr_curve_data.items():
    plt.plot(pr['recall'], pr['precision'], label=f"{model} (AUC = {pr['auc']:.2f})")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.tight_layout()
plt.savefig('data/pr_curve.png')
plt.close()

# Output the links to the plots that were saved
print("Classification Report Plot: data/classification_report.png")
print("ROC Curve Plot: data/roc_curve.png")
print("Precision-Recall Curve Plot: data/pr_curve.png")