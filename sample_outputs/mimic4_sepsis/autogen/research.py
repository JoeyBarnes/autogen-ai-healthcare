# filename: research.py

import json

# Relevant lab tests for predicting the onset of sepsis
relevant_lab_tests = [
    'aniongap',
    'bicarbonate',
    'bun',
    'creatinine',
    'glucose',
    'sodium',
    'potassium'
]

# Saving the findings to a JSON file
with open('data/lab_test_types.json', 'w') as file:
    json.dump(relevant_lab_tests, file)

print("Relevant lab test types for predicting sepsis have been saved to 'data/lab_test_types.json'.")