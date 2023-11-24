from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

# Initialize Spark Context and SQL Context
sc = SparkContext("local", "Distributed ELM Training")
sqlContext = SQLContext(sc)

# Load the data
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('C:\\Repo\\ELM_IDS\\CTU-13-Dataset\\8\\transformed_feature_vectors_2.csv')

# Convert the data to dense vector (features)
assembler = VectorAssembler(inputCols=data.columns, outputCol="features")
data = assembler.transform(data)

# Define the cost matrix C, regularization factor λ, and target labels T
C = np.diag(np.sqrt(data.select("features").rdd.flatMap(lambda x: x).collect()))
λ = 0.01
T = data.select("label").rdd.flatMap(lambda x: x).collect()

# Scale the values of the matrix H and vector T according to costs stored in the C matrix
Z = np.sqrt(C)
A = Z.dot(data.select("features").rdd.flatMap(lambda x: x).collect())
B = Z.dot(T)

# Calculate the matrix G = A^T x A
G = np.dot(A.T, A)

# Calculate right singular vectors V, singular values D, and left singular vectors U
U, D, V = np.linalg.svd(G)

# Calculate βˆ = Vx (D2 + λ)^(−1) x D x U^(T)x B
β = np.dot(V, np.dot(np.linalg.inv(D**2 + λ), np.dot(D, np.dot(U.T, B))))

print(β)