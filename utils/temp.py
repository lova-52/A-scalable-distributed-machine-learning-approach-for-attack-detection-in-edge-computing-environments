import pandas as pd
import numpy as np

# Define the path to the text file
text_file_path = "C:/Repo/ELM_IDS/CTU-13-Dataset/8/capture20110816-3.binetflow"

# Read the text file into a pandas DataFrame
data = pd.read_csv(text_file_path, sep=',', header=0, parse_dates=[0])

# Define the path for the output Excel file
excel_file_path = "C:/Repo/ELM_IDS/CTU-13-Dataset/8/capture20110816-3.xlsx"

# Split the data into multiple sheets if it's too large
chunk_size = 1048576  # Maximum number of rows per sheet
for i, chunk in enumerate(np.array_split(data, len(data) // chunk_size + 1)):
    sheet_name = f'Sheet_{i + 1}'
    chunk.to_excel(excel_file_path, sheet_name=sheet_name, index=False)

print(f"Data has been successfully saved to {excel_file_path}")
