// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.Vectors
// $example off$

object OraGaussianMixture {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("GaussianMixture")
    val sc = new SparkContext(conf)



    val data = sc.textFile("data/mllib/gmm_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()


    val gmm = new GaussianMixture().setK(2).run(parsedData)


    gmm.save(sc, "target/org/apache/spark/GaussianMixture/GaussianMixtureModel")
    val sameModel = GaussianMixtureModel.load(sc,
      "target/org/apache/spark/GaussianMixture/GaussianMixtureModel")


    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }


    sc.stop()
  }
}
// scalastyle:on println
