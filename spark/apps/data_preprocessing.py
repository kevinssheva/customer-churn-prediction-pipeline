from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder.appName('DataPreprocessing').getOrCreate()

# Load data
data = spark.read.csv('spark/data/WA_Fn-UseC_-Telco-Customer-Churn.csv', header=True, inferSchema=True)

# Clean data
data = data.fillna({'TotalCharges': 0}) \
           .withColumn('Churn', when(col('Churn') == 'Yes', 1).otherwise(0)) \
           .dropna()

# Save cleaned data
data.write.parquet('path/to/cleaned_data.parquet')
