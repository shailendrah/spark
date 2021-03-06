package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.Imputer
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * Run with:
 *   bin/run-example ml.OraImputer
 */
object OraImputer  {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Imputer")
      .getOrCreate()


    val df = spark.createDataFrame(Seq(
      (1.0, Double.NaN),
      (2.0, Double.NaN),
      (Double.NaN, 3.0),
      (4.0, 4.0),
      (5.0, 5.0)
    )).toDF("a", "b")

    val imputer = new Imputer()
      .setInputCols(Array("a", "b"))
      .setOutputCols(Array("out_a", "out_b"))

    val model = imputer.fit(df)
    model.transform(df).show()


    spark.stop()
  }
}
