package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.linalg.Vectors
// $example off$
import org.apache.spark.sql.SparkSession

object OraMaxAbsScaler  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("MaxAbsScaler")
      .getOrCreate()


    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features")

    val scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")


    val scalerModel = scaler.fit(dataFrame)


    val scaledData = scalerModel.transform(dataFrame)
    scaledData.select("features", "scaledFeatures").show()


    spark.stop()
  }
}
