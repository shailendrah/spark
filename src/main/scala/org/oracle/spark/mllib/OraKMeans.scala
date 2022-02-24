// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
// $example off$

object OraKMeans {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("KMeans")
    val sc = new SparkContext(conf)



    val data = sc.textFile("data/mllib/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()


    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)


    val WSSSE = clusters.computeCost(parsedData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")


    clusters.save(sc, "target/org/apache/spark/KMeans/KMeansModel")
    val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeans/KMeansModel")


    sc.stop()
  }
}
// scalastyle:on println
