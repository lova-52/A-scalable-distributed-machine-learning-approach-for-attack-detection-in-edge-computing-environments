from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.sql.functions import sqrt, col, monotonically_increasing_id
import numpy as np
import pandas as pd

def calculate_output_weights_spark(C, lambda_, H, T):
    # Initialize Spark Session
    spark = SparkSession.builder.appName("calculate_output_weights").getOrCreate()

    # Convert numpy arrays to Spark DataFrames
    C_spark = spark.createDataFrame([(float(c),) for c in np.nditer(C)], ["C"])
    H_spark = spark.createDataFrame([(float(h),) for h in np.nditer(H)], ["H"])
    T_spark = spark.createDataFrame([(float(t),) for t in np.nditer(T)], ["T"])

    # Add a common column for joining
    C_spark = C_spark.withColumn("id", monotonically_increasing_id())
    H_spark = H_spark.withColumn("id", monotonically_increasing_id())
    T_spark = T_spark.withColumn("id", monotonically_increasing_id())

    # Join the dataframes
    Z_spark = C_spark.join(H_spark, "id").join(T_spark, "id")

    # Calculate Z, A, and B
    Z_spark = Z_spark.withColumn("Z", sqrt("C"))
    A_spark = Z_spark.withColumn("A", col("Z") * col("H"))
    B_spark = Z_spark.withColumn("B", col("Z") * col("T"))

    # Step 3: Calculate G
    G_pandas = A_spark.toPandas().transpose().dot(A_spark.toPandas())
    G_spark = spark.createDataFrame(G_pandas)

    # Convert DataFrame to IndexedRowMatrix for SVD
    G_indexed = IndexedRowMatrix(G_spark.rdd.zipWithIndex().map(lambda x: IndexedRow(x[1], list(x[0]))))

    # Step 4: Calculate SVD of G
    svd = G_indexed.computeSVD(G_indexed.numCols(), computeU=True)
    U = svd.U
    s = svd.s
    V = svd.V

    # Convert singular values to DenseMatrix for operations
    s_matrix = DenseMatrix(len(s), len(s), np.diag(s).tolist())

    # Convert DenseMatrix and IndexedRowMatrix to numpy arrays for multiplication
    V_np = np.array(V.toArray())
    s_matrix_np = np.array(s_matrix.toArray())
    U_np = np.array(U.toBlockMatrix().toLocalMatrix().toArray())
    B_spark_np = np.array(B_spark.collect())

    # Step 5: Calculate beta_hat
    beta_hat = V_np.dot((s_matrix_np**2 + lambda_)**(-1)).dot(U_np.T).dot(B_spark_np.T)

    return beta_hat

# Initialize Spark Session
spark = SparkSession.builder.appName("ELM_IDS").getOrCreate()

# Read the CSV file
data_path = "C:/Repo/ELM_IDS/CTU-13-Dataset/8/transformed_feature_vectors_w_C.csv"
df = spark.read.csv(data_path, header=False, inferSchema=True)

# Split the DataFrame into features (H) and labels (T)
feature_columns = [f"_c{i}" for i in range(10)]  # Adjust the range if the number of feature columns changes
label_column = "_c10"  # Adjust if the position of the label column changes
cost_column = "_c11" # Adjust if the position of the Cost column changes

H_df = df.select(*feature_columns)
T_df = df.select(label_column)
C_df = df.select(cost_column)

# Convert DataFrames to numpy arrays
H = np.array(H_df.collect())
T = np.array(T_df.collect()).flatten()
C = np.array(C_df.collect()).flatten()

# Define C and lambda_ 
lambda_ = 0.422

# Call the function
beta_hat = calculate_output_weights_spark(C, lambda_, H, T)

# Specify the path where you want to save the beta_hat to a CSV file
output_file_path = "beta_hat.csv"

# Convert the beta_hat NumPy array to a Pandas DataFrame
beta_hat_df = pd.DataFrame(beta_hat)

# Save the Pandas DataFrame to a CSV file
beta_hat_df.to_csv(output_file_path, index=False)

# Optionally, you can also save the beta_hat to a NumPy binary file (.npy)
# np.save("beta_hat.npy", beta_hat)

print(f"beta_hat saved to {output_file_path}")