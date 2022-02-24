// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// $example on$
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
// $example off$

object OraTFIDF {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("TFIDF")
    val sc = new SparkContext(conf)



    val documents: RDD[Seq[String]] = sc.textFile("data/mllib/kmeans_data.txt")
      .map(_.split(" ").toSeq)

    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(documents)



    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)




    val idfIgnore = new IDF(minDocFreq = 2).fit(tf)
    val tfidfIgnore: RDD[Vector] = idfIgnore.transform(tf)


    println("tfidf: ")
    tfidf.collect.foreach(x => println(x))

    println("tfidfIgnore: ")
    tfidfIgnore.collect.foreach(x => println(x))

    sc.stop()
  }
}
// scalastyle:on println
