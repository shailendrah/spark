// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// $example on$
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
// $example off$

object OraPCAOnSourceVector {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("PCAOnSourceVector")
    val sc = new SparkContext(conf)


    val data: RDD[LabeledPoint] = sc.parallelize(Seq(
      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 1)),
      new LabeledPoint(1, Vectors.dense(1, 1, 0, 1, 0)),
      new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0)),
      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 0)),
      new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0))))


    val pca = new PCA(5).fit(data.map(_.features))



    val projected = data.map(p => p.copy(features = pca.transform(p.features)))

    val collect = projected.collect()
    println("Projected vector of principal component:")
    collect.foreach { vector => println(vector) }

    sc.stop()
  }
}
// scalastyle:on println
