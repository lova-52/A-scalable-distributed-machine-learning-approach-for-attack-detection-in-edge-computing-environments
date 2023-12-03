import pandas as pd

# Define the path to the Excel file
excel_file_path = "C:/Repo/ELM_IDS/CTU-13-Dataset/8/capture20110816-3.xlsx"

# Read the Excel file into a pandas DataFrame
data = pd.read_excel(excel_file_path)

# Get unique values from the 'Label' column
unique_labels = data['Label'].unique()

# Convert the unique values to a list if needed
unique_labels_list = unique_labels.tolist()

print(unique_labels_list)
