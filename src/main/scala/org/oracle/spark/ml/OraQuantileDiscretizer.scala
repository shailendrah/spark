package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.QuantileDiscretizer
// $example off$
import org.apache.spark.sql.SparkSession

object OraQuantileDiscretizer  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("QuantileDiscretizer")
      .getOrCreate()


    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
    val df = spark.createDataFrame(data).toDF("id", "hour")




      .repartition(1)


    val discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3)

    val result = discretizer.fit(df).transform(df)
    result.show(false)


    spark.stop()
  }
}
