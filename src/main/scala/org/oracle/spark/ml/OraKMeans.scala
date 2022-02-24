package org.oracle.spark.ml

// scalastyle:off println

// $example on$
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * Run with
 * {{{
 * bin/run-example ml.OraKMeans
 * }}}
 */
object OraKMeans  {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()



    val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")


    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)


    val predictions = model.transform(dataset)


    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")


    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)


    spark.stop()
  }
}
// scalastyle:on println
