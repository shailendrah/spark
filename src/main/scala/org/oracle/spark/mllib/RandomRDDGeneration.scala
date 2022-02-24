// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD

/**
 * bin/run-example RandomRDDGeneration
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object RandomRDDGeneration {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName(s"RandomRDDGeneration")
    val sc = new SparkContext(conf)

    val numExamples = 10000
    val fraction = 0.1


    val normalRDD: RDD[Double] = RandomRDDs.normalRDD(sc, numExamples)
    println(s"Generated RDD of ${normalRDD.count()}" +
      " examples sampled from the standard normal distribution")
    println("  First 5 samples:")
    normalRDD.take(5).foreach( x => println(s"    $x") )


    val normalVectorRDD = RandomRDDs.normalVectorRDD(sc, numRows = numExamples, numCols = 2)
    println(s"Generated RDD of ${normalVectorRDD.count()} examples of length-2 vectors.")
    println("  First 5 samples:")
    normalVectorRDD.take(5).foreach( x => println(s"    $x") )

    println()

    sc.stop()
  }

}
// scalastyle:on println
