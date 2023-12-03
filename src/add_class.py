import pandas as pd

# Load the datasets into pandas DataFrames
df1 = pd.read_excel('C:/Repo/ELM_IDS/ELM_IDS/ELM_IDS/dataset/capture20110816-3_no_empty_rows.xlsx')
df2 = pd.read_excel('C:/Repo/ELM_IDS/ELM_IDS/ELM_IDS/dataset/feature_vectors.xlsx')

# Create a dictionary that maps SrcAddr to Class from the first DataFrame
class_mapping = df1.set_index('SrcAddr')['Class'].to_dict()

# Map the SrcAddr to Class in the second DataFrame using the dictionary
# If the SrcAddr is not found in the dictionary, it will assign NaN
df2['Class'] = df2['SrcAddr'].map(class_mapping)

# If there are any NaN values that were not found in the mapping,
# you might want to set them to a default value, e.g., 0 or another class.
df2['Class'] = df2['Class'].fillna(0)

# Save the updated DataFrame back to an Excel file
df2.to_excel('C:/Repo/ELM_IDS/ELM_IDS/ELM_IDS/dataset/feature_vectors_with_class.xlsx', index=False)
