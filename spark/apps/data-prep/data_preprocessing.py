from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

def process_data(input_path: str, experiment_id: str, numeric_columns: list) -> None:
    spark = SparkSession.builder \
        .appName("CustomerChurnProcessing") \
        .config("spark.jars.packages", "org.mlflow:mlflow-spark:2.19.0") \
        .getOrCreate()

    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Data manipulation
    df = df.drop("customerID")
    
    # Convert TotalCharges to double and handle null values
    df = df.withColumn("TotalCharges", F.col("TotalCharges").cast(DoubleType()))
    avg_total_charges = df.select(F.avg("TotalCharges")).first()[0]
    df = df.fillna(avg_total_charges, subset=["TotalCharges"])
    
    # Filter out records where tenure is 0
    df = df.filter(F.col("tenure") != 0)
    
    # Convert SeniorCitizen to Yes/No
    df = df.withColumn("SeniorCitizen", 
                      F.when(F.col("SeniorCitizen") == 1, "Yes")
                       .otherwise("No"))
    
    # Get categorical columns (excluding numeric columns)
    categorical_columns = [field.name for field in df.schema.fields 
                         if field.dataType.simpleString() == "string"]

    # Create pipeline stages
    stages = []
    
    # String Indexing for categorical columns
    for column in categorical_columns:
        indexer = StringIndexer(
            inputCol=column,
            outputCol=f"{column}_indexed",
            handleInvalid="keep"
        )
        stages.append(indexer)
    
    # Vector Assembly for numeric features
    numeric_assembler = VectorAssembler(
        inputCols=numeric_columns,
        outputCol="numeric_features",
        handleInvalid="keep"
    )
    stages.append(numeric_assembler)
    
    # Standard Scaler for numeric features
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="scaled_numeric_features",
        withStd=True,
        withMean=True
    )
    stages.append(scaler)
    
    # Create and fit pipeline
    pipeline = Pipeline(stages=stages)
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name="Spark_Processing") as run:
        # Fit and transform the data
        model = pipeline.fit(df)
        processed_df = model.transform(df)
        
        # Log the pipeline model
        mlflow.spark.log_model(model, "preprocessing_pipeline")
        
        # Log metrics
        numeric_stats = processed_df.select([
            F.mean(F.col("scaled_numeric_features")).alias("mean_scaled_features"),
            F.stddev(F.col("scaled_numeric_features")).alias("std_scaled_features")
        ]).first()
        
        mlflow.log_metrics({
            "mean_scaled_features": float(numeric_stats["mean_scaled_features"]),
            "std_scaled_features": float(numeric_stats["std_scaled_features"])
        })
        
        # Select relevant columns for final dataset
        final_columns = [f"{col}_indexed" for col in categorical_columns]
        final_columns.append("scaled_numeric_features")
        
        final_df = processed_df.select(final_columns)
        
        return final_df

if __name__ == "__main__":
    # Example usage
    NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
    FILE_PATH = "spark/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    EXPERIMENT_ID = "your_experiment_id"
    
    processed_df = process_data(
        input_path=FILE_PATH,
        experiment_id=EXPERIMENT_ID,
        numeric_columns=NUMERIC_COLUMNS
    )