// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.regression.IsotonicRegression
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * Run with
 * {{{
 * bin/run-example ml.OraIsotonicRegression
 * }}}
 */
object OraIsotonicRegression  {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()



    val dataset = spark.read.format("libsvm")
      .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")


    val ir = new IsotonicRegression()
    val model = ir.fit(dataset)

    println(s"Boundaries in increasing order: ${model.boundaries}\n")
    println(s"Predictions associated with the boundaries: ${model.predictions}\n")


    model.transform(dataset).show()


    spark.stop()
  }
}
// scalastyle:on println
