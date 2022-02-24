package org.oracle.spark.ml

// scalastyle:off println

// $example on$
import org.apache.spark.ml.clustering.GaussianMixture
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * Run with
 * {{{
 * bin/run-example ml.OraGaussianMixture
 * }}}
 */
object OraGaussianMixture  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
        .builder
        .appName(s"${this.getClass.getSimpleName}")
        .getOrCreate()



    val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")


    val gmm = new GaussianMixture()
      .setK(2)
    val model = gmm.fit(dataset)


    for (i <- 0 until model.getK) {
      println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
          s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
    }


    spark.stop()
  }
}
// scalastyle:on println
