// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
// $example off$

object OraPMMLModelExport {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("PMMLModelExport")
    val sc = new SparkContext(conf)



    val data = sc.textFile("data/mllib/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()


    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)


    println(s"PMML Model:\n ${clusters.toPMML}")


    clusters.toPMML("/tmp/kmeans.xml")


    clusters.toPMML(sc, "/tmp/kmeans")


    clusters.toPMML(System.out)


    sc.stop()
  }
}
// scalastyle:on println
