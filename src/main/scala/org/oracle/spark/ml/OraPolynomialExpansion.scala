// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors
// $example off$
import org.apache.spark.sql.SparkSession

object OraPolynomialExpansion  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("PolynomialExpansion")
      .getOrCreate()


    val data = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val polyExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3)

    val polyDF = polyExpansion.transform(df)
    polyDF.show(false)


    spark.stop()
  }
}
// scalastyle:on println
