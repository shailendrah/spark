// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
// $example off$

object OraSummaryStatistics {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("SummaryStatistics")
    val sc = new SparkContext(conf)


    val observations = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(3.0, 30.0, 300.0)
      )
    )


    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean)
    println(summary.variance)
    println(summary.numNonzeros)


    sc.stop()
  }
}
// scalastyle:on println
