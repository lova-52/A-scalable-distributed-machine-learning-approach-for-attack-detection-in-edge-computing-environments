import pandas as pd

# Read the dataset file into a pandas DataFrame
df = pd.read_excel('C:/Repo/ELM_IDS/ELM_IDS/ELM_IDS/dataset/capture20110816-3_no_empty_rows.xlsx')

# Convert the StartTime column to datetime format
df['StartTime'] = pd.to_datetime(df['StartTime'])

# Define the time window durations in minutes
time_windows = [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2)]

# Function to calculate statistics for a given group of NetFlows
def calculate_statistics(group_df):
    num_flows = len(group_df)
    sum_bytes = group_df['TotBytes'].sum()
    avg_bytes_per_flow = group_df['TotBytes'].mean() if num_flows > 0 else 0
    avg_communication_time = group_df['Dur'].mean() if num_flows > 0 else 0
    num_unique_ips = group_df['DstAddr'].nunique()
    num_unique_ports = group_df['Dport'].nunique()
    num_unique_protocols = group_df['Proto'].nunique()

    return [num_flows, sum_bytes, avg_bytes_per_flow, avg_communication_time,
            num_unique_ips, num_unique_ports, num_unique_protocols]

# Initialize an empty list to store the feature vectors
feature_vectors = []

# Iterate over each time window
for time_window in time_windows:
    # Group the NetFlows by time window and IP source address
    grouped = df.groupby([pd.Grouper(key='StartTime', freq=time_window), 'SrcAddr'])
    
    # Iterate over the groups and calculate the statistics for each group
    for group_name, group_df in grouped:
        time_window_start = group_name[0]
        ip_address = group_name[1]
        
        # Calculate statistics for the group
        stats = calculate_statistics(group_df)

        # Create a feature vector and append it to the list
        feature_vector = [time_window_start, ip_address] + stats
        feature_vectors.append(feature_vector)

# Define the columns for the DataFrame
columns = ['TimeWindowStart', 'SrcAddr', 'NumFlows', 'SumBytes', 'AvgBytesPerFlow',
           'AvgCommunicationTime', 'NumUniqueIPs', 'NumUniquePorts', 'NumUniqueProtocols']

# Convert the list of feature vectors to a pandas DataFrame
feature_vectors_df = pd.DataFrame(feature_vectors, columns=columns)

# Define the path for the output Excel file for the feature vectors
output_feature_vectors_excel_file = 'C:/Repo/ELM_IDS/ELM_IDS/ELM_IDS/dataset/feature_vectors.xlsx'

# Save the feature vectors to an Excel file
feature_vectors_df.to_excel(output_feature_vectors_excel_file, index=False)

print(f"Feature vectors have been saved to {output_feature_vectors_excel_file}")
