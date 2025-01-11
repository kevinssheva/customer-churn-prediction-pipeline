from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName('EvaluateModel').getOrCreate()

# Load test data
data = spark.read.parquet('path/to/features_data.parquet')

# Load model
model = PipelineModel.load('path/to/saved_model')

# Evaluate model
predictions = model.transform(data)
evaluator = BinaryClassificationEvaluator(labelCol='Churn', metricName='areaUnderROC')
roc_auc = evaluator.evaluate(predictions)

print(f"Model ROC AUC: {roc_auc}")
