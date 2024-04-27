# filename: research.py
import json

# List of lab test types relevant to Acute Kidney Injury
lab_test_types = [
    "creatinine",
    "bun",
    "potassium",
    "sodium",
    "bicarbonate"
]

# Save the findings to 'data/lab_test_types.json'
with open('data/lab_test_types.json', 'w') as file:
    json.dump(lab_test_types, file)

print("Lab test types relevant to Acute Kidney Injury have been saved to 'data/lab_test_types.json'")