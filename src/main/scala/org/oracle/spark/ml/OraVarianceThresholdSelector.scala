// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.VarianceThresholdSelector
import org.apache.spark.ml.linalg.Vectors
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * Run with
 * {{{
 * bin/run-example ml.OraVarianceThresholdSelector
 * }}}
 */
object OraVarianceThresholdSelector  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("VarianceThresholdSelector")
      .getOrCreate()
    import spark.implicits._


    val data = Seq(
      (1, Vectors.dense(6.0, 7.0, 0.0, 7.0, 6.0, 0.0)),
      (2, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0)),
      (3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0)),
      (4, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0)),
      (5, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0)),
      (6, Vectors.dense(8.0, 9.0, 6.0, 0.0, 0.0, 0.0))
    )

    val df = spark.createDataset(data).toDF("id", "features")

    val selector = new VarianceThresholdSelector()
      .setVarianceThreshold(8.0)
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures")

    val result = selector.fit(df).transform(df)

    println(s"Output: Features with variance lower than" +
      s" ${selector.getVarianceThreshold} are removed.")
    result.show()


    spark.stop()
  }
}
// scalastyle:on println
