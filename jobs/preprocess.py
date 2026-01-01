from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
import os

def main():
    print("=" * 80)
    print("Step 1: Spark Weather Data Preprocessing")
    print("=" * 80)
    
    spark = SparkSession.builder \
        .appName("WeatherPreprocessing") \
        .getOrCreate()
    
    # 1. Load Real Data
    input_path = "/opt/spark/data/raw/weather.csv"
    print(f"Reading data from: {input_path}")
    
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df.show()
    
    # 2. Encode Labels (rain=1, no rain=0)
    print("Encoding target variable 'Rain'...")
    df = df.withColumn("label_encoded", 
        when(col("Rain") == "rain", 1).otherwise(0)
    )
    
    # 3. Normalize Features
    # We have 5 input features
    feature_cols = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]
    
    print("Normalizing features...")
    for col_name in feature_cols:
        stats = df.select(mean(col(col_name)).alias('mean'), stddev(col(col_name)).alias('std')).first()
        
        # Avoid division by zero
        std_val = stats['std'] if stats['std'] != 0 else 1.0
        
        df = df.withColumn(
            f"{col_name}_norm",
            (col(col_name) - stats['mean']) / std_val
        )

    # 4. Save to Parquet
    output_path = "/opt/spark/data/processed/data.parquet"
    print(f"Saving processed data to: {output_path}")
    
    # Select only normalized columns and the label
    final_cols = [f"{c}_norm" for c in feature_cols] + ["label_encoded"]
    final_df = df.select(final_cols)
    
    final_df.write.mode("overwrite").parquet(output_path)
    print("✓ Preprocessing Complete!")
    spark.stop()

if __name__ == "__main__":
    main()