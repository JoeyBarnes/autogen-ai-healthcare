# filename: visualization.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model performance details into a pandas dataframe
with open('data/model_details.json', 'r') as file:
    model_details = json.load(file)

# Prepare data for the classification report bar plot
cr_data = {}
for model_name, details in model_details.items():
    cr = details['classification_report']['weighted avg']
    accuracy = details['classification_report']['accuracy']
    cr_data[model_name.replace("Classifier", "")] = {
        'Precision': cr['precision'],
        'Recall': cr['recall'],
        'F1-Score': cr['f1-score'],
        'Accuracy': accuracy
    }

# Convert cr_data to DataFrame for plotting
cr_df = pd.DataFrame(cr_data).T.reset_index().melt(id_vars="index")
cr_df.columns = ['Model', 'Performance Metric', 'Value']

# Plotting the classification report data
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Value', hue='Performance Metric', data=cr_df)
plt.ylim(min(cr_df['Value']) * 0.95, max(cr_df['Value']) * 1.05)
plt.xticks(rotation=45)
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('data/classification_report.png')
plt.close()

# Prepare data for the ROC curve plot
roc_curve_data = {model_name.replace("Classifier", ""): details['roc_curve'] for model_name, details in model_details.items()}

# Plotting the ROC curve
plt.figure(figsize=(10, 6))
for model_name, roc_data in roc_curve_data.items():
    plt.plot(roc_data['fpr'], roc_data['tpr'], label=f'{model_name} (AUC = {roc_data["auc"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('data/roc_curve.png')
plt.close()

# Prepare data for the precision-recall curve plot
pr_curve_data = {model_name.replace("Classifier", ""): details['pr_curve'] for model_name, details in model_details.items()}

# Plotting the Precision-Recall curve
plt.figure(figsize=(10, 6))
for model_name, pr_data in pr_curve_data.items():
    plt.plot(pr_data['recall'], pr_data['precision'], label=f'{model_name} (AUC = {pr_data["auc"]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.savefig('data/pr_curve.png')
plt.close()

# Output the links to the plots
print("Links to the plots:")
print("Classification Report: data/classification_report.png")
print("ROC Curve: data/roc_curve.png")
print("Precision-Recall Curve: data/pr_curve.png")