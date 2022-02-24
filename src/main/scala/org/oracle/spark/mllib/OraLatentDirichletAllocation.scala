// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.Vectors
// $example off$

object OraLatentDirichletAllocation {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("LatentDirichletAllocation")
    val sc = new SparkContext(conf)



    val data = sc.textFile("data/mllib/sample_lda_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))

    val corpus = parsedData.zipWithIndex.map(_.swap).cache()


    val ldaModel = new LDA().setK(3).run(corpus)


    println(s"Learned topics (as distributions over vocab of ${ldaModel.vocabSize} words):")
    val topics = ldaModel.topicsMatrix
    for (topic <- Range(0, 3)) {
      print(s"Topic $topic :")
      for (word <- Range(0, ldaModel.vocabSize)) {
        print(s"${topics(word, topic)}")
      }
      println()
    }


    ldaModel.save(sc, "target/org/apache/spark/LatentDirichletAllocation/LDAModel")
    val sameModel = DistributedLDAModel.load(sc,
      "target/org/apache/spark/LatentDirichletAllocation/LDAModel")


    sc.stop()
  }
}
// scalastyle:on println
