import pandas as pd
import numpy as np

# Load the data from the excel file
df = pd.read_excel('C:\\Repo\\ELM_IDS\\CTU-13-Dataset\\8\\feature_vectors.xlsx')

# Normalize the input data using min-max scaling
min_vals = df.min()
max_vals = df.max()
df = (df - min_vals) / (max_vals - min_vals)
print(df)

# Generate a random projection matrix and bias vector
num_features = df.shape[1]
num_hidden_neurons = 10  # Choose the desired number of neurons in the hidden layer

a = np.random.rand(num_features, num_hidden_neurons)
b = np.random.rand(num_hidden_neurons)

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df_transformed = df.apply(lambda x: sigmoid(np.dot(x, a) + b), axis=1, result_type='expand')

#Save the transformed dataframe to a new excel file
df_transformed.to_excel('C:\\Repo\\ELM_IDS\\CTU-13-Dataset\\8\\transformed_feature_vectors.xlsx', index=False)
