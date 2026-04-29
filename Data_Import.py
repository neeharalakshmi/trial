from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
import pandas as pd
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.appName("TrojanDetection").getOrCreate()

# Load dataset using Pandas
file_path = "/kaggle/input/trojan-detection/Trojan_Detection.csv"  # Update with correct path
try:
    df_pandas = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading file: {e}")

# Rename columns to column1, column2, ...
df_pandas.columns = [f"column{i+1}" if col != "Class" else "Class" for i, col in enumerate(df_pandas.columns)]

# Convert Pandas data types to PySpark-compatible types (except 'Class' column)
def infer_spark_type(col_name, dtype):
    if col_name == "Class":  # Keep "Class" as StringType
        return StringType()
    if np.issubdtype(dtype, np.integer):
        return IntegerType()
    elif np.issubdtype(dtype, np.floating):
        return DoubleType()
    else:
        return StringType()

# Define schema for PySpark DataFrame
schema = StructType([
    StructField(col, infer_spark_type(col, df_pandas[col].dtype), True) for col in df_pandas.columns
])
# Drop NaN and Inf values
df_pandas.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf with NaN
df_pandas.dropna(inplace=True)  # Drop rows with NaN
# Convert Pandas DataFrame to PySpark DataFrame
df = spark.createDataFrame(df_pandas, schema=schema)

# Show the PySpark DataFrame
df.show(5)
df.printSchema()
# Get all string columns
string_cols = [col_name for col_name, dtype in df.dtypes if dtype == "string"]

# Print unique values in string columns to find issues
for col_name in string_cols:
    print(f"Unique values in {col_name}:")
    df.select(col_name).distinct().show(10)
df.groupBy("Class").count().show()  # Check class distribution
[(col.name, col.dataType) for col in df.schema]
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

# Identify numerical columns (excluding 'Class' and other non-numeric columns)
numeric_cols = [col_name for col_name, dtype in df.dtypes if dtype in ["int", "double"] and col_name != "Class"]

# Identify categorical columns
categorical_cols = [col_name for col_name, dtype in df.dtypes if dtype == "string" and col_name != "Class"]

# Convert categorical columns to numerical using StringIndexer
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed") for col_name in categorical_cols]
for indexer in indexers:
    df = indexer.fit(df).transform(df)

# Add indexed categorical columns to numeric columns list
indexed_categorical_cols = [col_name + "_indexed" for col_name in categorical_cols]
final_numeric_cols = numeric_cols + indexed_categorical_cols

# Assemble numeric features into a single vector column
assembler = VectorAssembler(inputCols=final_numeric_cols, outputCol="features")
df = assembler.transform(df)

# Define StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# Fit and transform the DataFrame
scaler_model = scaler.fit(df)
df_scaled = scaler_model.transform(df)

# Convert vector column to separate columns
df_scaled = df_scaled.withColumn("scaled_features", vector_to_array("scaled_features"))

# Split the vector column into separate numerical columns
for i, col_name in enumerate(final_numeric_cols):
    df_scaled = df_scaled.withColumn(col_name + "_scaled", df_scaled["scaled_features"][i])

# Drop unnecessary columns
df_scaled = df_scaled.drop("features", "scaled_features")

# Show the results
df_scaled.show()
from pyspark.ml.feature import VectorAssembler

# Identify all scaled feature columns
scaled_feature_cols = [col_name for col_name in df_scaled.columns if col_name.endswith("_scaled")]

# Reassemble the scaled features into a single vector column
feature_assembler = VectorAssembler(inputCols=scaled_feature_cols, outputCol="scaled_features")
df_scaled = feature_assembler.transform(df_scaled)
from pyspark.ml.feature import StringIndexer

# Convert the 'Class' column from string to numeric
indexer = StringIndexer(inputCol="Class", outputCol="label")
df_scaled = indexer.fit(df_scaled).transform(df_scaled)

# Drop the original 'Class' column if necessary
df_scaled = df_scaled.drop("Class")

# Verify the transformation
df_scaled.select("label").show(5)
# Perform train-test split
train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)

# Show dataset sizes
print(f"Train Size: {train_df.count()}, Test Size: {test_df.count()}")
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define Random Forest Model
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", numTrees=100)

# Train the model
rf_model = rf.fit(train_df)

# Make predictions on the test set
rf_predictions = rf_model.transform(test_df)

# Evaluate the model
rf_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_accuracy = rf_evaluator.evaluate(rf_predictions)

print(f"Random Forest Accuracy: {rf_accuracy}")
