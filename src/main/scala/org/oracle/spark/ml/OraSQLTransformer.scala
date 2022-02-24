// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.SQLTransformer
// $example off$
import org.apache.spark.sql.SparkSession

object OraSQLTransformer  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("SQLTransformer")
      .getOrCreate()


    val df = spark.createDataFrame(
      Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")

    val sqlTrans = new SQLTransformer().setStatement(
      "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

    sqlTrans.transform(df).show()


    spark.stop()
  }
}
// scalastyle:on println
