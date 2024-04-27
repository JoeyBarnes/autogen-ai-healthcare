# filename: research.py
import json

# List of lab test types relevant for predicting the onset of Sepsis
relevant_lab_tests = [
    'aniongap',
    'bicarbonate',
    'bun',
    'creatinine',
    'glucose',
    'sodium',
    'potassium'
]

# Save the findings to a JSON file
with open('data/lab_test_types.json', 'w') as file:
    json.dump(relevant_lab_tests, file)

print("Relevant lab test types for predicting Sepsis have been saved to 'data/lab_test_types.json'.")