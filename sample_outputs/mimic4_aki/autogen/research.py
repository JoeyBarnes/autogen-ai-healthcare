# filename: research.py
import json

# Define the lab test types relevant to Acute Kidney Injury
aki_relevant_tests = [
    'bun',
    'creatinine',
    'potassium',
    'sodium',
    'bicarbonate'
]

# Specify the file path
file_path = 'data/lab_test_types.json'

# Ensure the directory exists
import os
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Write the relevant lab test types to the file
with open(file_path, 'w') as file:
    json.dump(aki_relevant_tests, file)

print(f"Lab test types relevant to Acute Kidney Injury have been saved to {file_path}.")