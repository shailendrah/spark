package org.oracle.spark.mllib

// scalastyle:off println
import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
// $example off$

/**
 * bin/run-example mllib.OraBisectingKMeans
 */
object OraBisectingKMeans {

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("mllib.BisectingKMeans")
    val sc = new SparkContext(sparkConf)



    def parse(line: String): Vector = Vectors.dense(line.split(" ").map(_.toDouble))
    val data = sc.textFile("data/mllib/kmeans_data.txt").map(parse).cache()


    val bkm = new BisectingKMeans().setK(6)
    val model = bkm.run(data)


    println(s"Compute Cost: ${model.computeCost(data)}")
    model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
      println(s"Cluster Center ${idx}: ${center}")
    }


    sc.stop()
  }
}
// scalastyle:on println
