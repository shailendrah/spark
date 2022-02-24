package org.oracle.spark.ml

// scalastyle:off println

// $example on$
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

object OraBisectingKMeans  {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("BisectingKMeans")
      .getOrCreate()



    val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")


    val bkm = new BisectingKMeans().setK(2).setSeed(1)
    val model = bkm.fit(dataset)


    val predictions = model.transform(dataset)


    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")


    println("Cluster Centers: ")
    val centers = model.clusterCenters
    centers.foreach(println)


    spark.stop()
  }
}
// scalastyle:on println

