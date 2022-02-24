// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.RFormula
// $example off$
import org.apache.spark.sql.SparkSession

object OraRFormula  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RFormula")
      .getOrCreate()


    val dataset = spark.createDataFrame(Seq(
      (7, "US", 18, 1.0),
      (8, "CA", 12, 0.0),
      (9, "NZ", 15, 0.0)
    )).toDF("id", "country", "hour", "clicked")

    val formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("features")
      .setLabelCol("label")

    val output = formula.fit(dataset).transform(dataset)
    output.select("features", "label").show()


    spark.stop()
  }
}
// scalastyle:on println
