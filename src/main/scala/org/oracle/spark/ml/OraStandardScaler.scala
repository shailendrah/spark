// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.StandardScaler
// $example off$
import org.apache.spark.sql.SparkSession

object OraStandardScaler  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("StandardScaler")
      .getOrCreate()


    val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)


    val scalerModel = scaler.fit(dataFrame)


    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()


    spark.stop()
  }
}
// scalastyle:on println
