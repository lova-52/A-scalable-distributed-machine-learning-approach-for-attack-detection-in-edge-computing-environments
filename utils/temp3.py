import pandas as pd

# Define the path to the Excel file
excel_file_path = "C:/Repo/ELM_IDS/CTU-13-Dataset/8/capture20110816-3.xlsx"

# Read the Excel file into a pandas DataFrame
data = pd.read_excel(excel_file_path)

# Remove rows with empty values in any column
data = data.dropna()

# Define the path for the output Excel file after removing empty rows
output_excel_file_path = "C:/Repo/ELM_IDS/CTU-13-Dataset/8/capture20110816-3_no_empty_rows.xlsx"

# Save the cleaned data to a new Excel file
data.to_excel(output_excel_file_path, index=False)

print(f"Rows with empty values have been removed. Cleaned data saved to {output_excel_file_path}")
