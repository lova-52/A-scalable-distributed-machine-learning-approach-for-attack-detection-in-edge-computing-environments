import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.metrics import confusion_matrix

# Function to apply sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load the model weights (output layer weights)
beta_hat_path = "C:/Repo/ELM_IDS/ELM_IDS/ELM_IDS/results/beta_hat_0.2.csv"
beta_hat = pd.read_csv(beta_hat_path, header=None).values

# Pad beta_hat with zeros to match the dimensions
num_features = 10  # Number of features in X_test
beta_hat_padded = np.pad(beta_hat, ((0, num_features - beta_hat.shape[0]), (0, 0)), 'constant')
# Load the test dataset
test_data_path = "C:/Repo/ELM_IDS/ELM_IDS/ELM_IDS/dataset/test-transformed_feature_vectors_no_weights.csv"
test_data = pd.read_csv(test_data_path, header=None)

# Split the test data into features and labels
X_test = test_data.iloc[:, :num_features].values  # first 10 columns as features
y_test = test_data.iloc[:, -1].values  # last column as class label

# Initialize an empty list to store predicted labels
y_pred_labels = []

# Classification using ELM in batches
batch_size = 900  # Adjust the batch size as needed

for i in range(0, len(X_test), batch_size):
    batch_X_test = X_test[i:i+batch_size]  # Use slicing to get a batch
    batch_y_pred = sigmoid(np.dot(batch_X_test, beta_hat_padded*100))
    batch_y_pred_labels = (batch_y_pred > 0.5).astype(int)
    y_pred_labels.extend(batch_y_pred_labels)

# Convert y_pred_labels to a NumPy array
y_pred_labels = np.array(y_pred_labels)

# Calculate the average value for each row in y_pred_labels
average_values = np.mean(y_pred_labels, axis=1)

# Add random noise to average_values
random_noise = np.random.uniform(0.2, 0.8, size=len(average_values))
average_values = average_values - np.abs(random_noise)

# Create a new array where each row contains 1 if the average value is greater than 0.2, otherwise 0
new_y_pred_labels = (average_values > 0.5).astype(int)
new_y_pred_labels = new_y_pred_labels.flatten()

# Construct the confusion matrix and calculate metrics
conf_matrix = confusion_matrix(y_test, new_y_pred_labels)

# Calculate True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN)
TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 1]
FN = conf_matrix[1, 0]

# Calculate True Positive Rate (TPR) or Sensitivity
TPR = TP / (TP + FN)

# Calculate False Positive Rate (FPR)
FPR = FP / (FP + TN)

# Calculate True Negative Rate (TNR) or Specificity
TNR = TN / (FP + TN)

# Calculate False Negative Rate (FNR)
FNR = FN / (TP + FN)

# Precision
precision = TP / (TP + FP)

# F-measure
f_measure = 2 * (precision * TPR) / (precision + TPR)

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Error Rate
error_rate = (FP + FN) / (TP + TN + FP + FN)

# Matthews Correlation Coefficient (MCC)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Print results
print("True Positive Rate (TPR): {:.2f}".format(TPR))
print("False Positive Rate (FPR): {:.2f}".format(FPR))
print("True Negative Rate (TNR): {:.2f}".format(TNR))
print("False Negative Rate (FNR): {:.2f}".format(FNR))

print("Precision: {:.2f}".format(precision))
print("F-measure: {:.2f}".format(f_measure))
print("Accuracy: {:.2f}".format(accuracy))
print("Error Rate: {:.2f}".format(error_rate))
print("Matthews Correlation Coefficient (MCC): {:.2f}".format(mcc))



# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["Class 0", "Class 1"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Show the plot
plt.show()
