// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
// $example off$

object OraCorrelations {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Correlations")
    val sc = new SparkContext(conf)


    val seriesX: RDD[Double] = sc.parallelize(Array(1, 2, 3, 3, 5))

    val seriesY: RDD[Double] = sc.parallelize(Array(11, 22, 33, 33, 555))



    val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")
    println(s"Correlation is: $correlation")

    val data: RDD[Vector] = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(5.0, 33.0, 366.0))
    )



    val correlMatrix: Matrix = Statistics.corr(data, "pearson")
    println(correlMatrix.toString)


    sc.stop()
  }
}
// scalastyle:on println

