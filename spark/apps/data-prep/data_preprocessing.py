import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Set environment variables (if not already set in your system)
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

# Initialize Spark session with MinIO configurations
spark = SparkSession.builder \
    .appName('DataPreprocessing') \
    .config('spark.hadoop.fs.s3a.endpoint', 'http://localhost:9001') \
    .config('spark.hadoop.fs.s3a.access.key', os.getenv('AWS_ACCESS_KEY_ID')) \
    .config('spark.hadoop.fs.s3a.secret.key', os.getenv('AWS_SECRET_ACCESS_KEY')) \
    .config('spark.hadoop.fs.s3a.path.style.access', 'true') \
    .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') \
    .getOrCreate()

# Load data
data = spark.read.csv('/opt/spark-data/WA_Fn-UseC_-Telco-Customer-Churn.csv', header=True, inferSchema=True)

# Clean data
data = data.fillna({'TotalCharges': 0}) \
           .withColumn('Churn', when(col('Churn') == 'Yes', 1).otherwise(0)) \
           .dropna()

# Save cleaned data to MinIO
data.write.parquet('s3a://mlops/cleaned_data.parquet')
