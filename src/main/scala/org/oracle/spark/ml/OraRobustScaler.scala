// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.RobustScaler
// $example off$
import org.apache.spark.sql.SparkSession

object OraRobustScaler  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RobustScaler")
      .getOrCreate()


    val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val scaler = new RobustScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithScaling(true)
      .setWithCentering(false)
      .setLower(0.25)
      .setUpper(0.75)


    val scalerModel = scaler.fit(dataFrame)


    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()


    spark.stop()
  }
}
// scalastyle:on println
