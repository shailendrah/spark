// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.Binarizer
// $example off$
import org.apache.spark.sql.SparkSession

object OraBinarizer  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("Binarizer")
      .getOrCreate()


    val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
    val dataFrame = spark.createDataFrame(data).toDF("id", "feature")

    val binarizer: Binarizer = new Binarizer()
      .setInputCol("feature")
      .setOutputCol("binarized_feature")
      .setThreshold(0.5)

    val binarizedDataFrame = binarizer.transform(dataFrame)

    println(s"Binarizer output with Threshold = ${binarizer.getThreshold}")
    binarizedDataFrame.show()


    spark.stop()
  }
}
// scalastyle:on println
