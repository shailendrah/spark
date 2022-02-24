// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.classification.LinearSVC
// $example off$
import org.apache.spark.sql.SparkSession

object OraLinearSVC  {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearSVC")
      .getOrCreate()



    val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)


    val lsvcModel = lsvc.fit(training)


    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")


    spark.stop()
  }
}
// scalastyle:on println
