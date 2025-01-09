from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.appName('FeatureEngineering').getOrCreate()

# Load cleaned data
data = spark.read.parquet('path/to/cleaned_data.parquet')

# Encode categorical variables
categorical_columns = ['gender', 'Partner', 'Dependents']
for col_name in categorical_columns:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
    data = indexer.fit(data).transform(data)

# Save features
data.write.parquet('path/to/features_data.parquet')
