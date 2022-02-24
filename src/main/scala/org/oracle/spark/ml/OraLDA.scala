package org.oracle.spark.ml

// scalastyle:off println
// $example on$
import org.apache.spark.ml.clustering.LDA
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * Run with
 * {{{
 * bin/run-example ml.OraLDA
 * }}}
 */
object OraLDA  {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()



    val dataset = spark.read.format("libsvm")
      .load("data/mllib/sample_lda_libsvm_data.txt")


    val lda = new LDA().setK(10).setMaxIter(10)
    val model = lda.fit(dataset)

    val ll = model.logLikelihood(dataset)
    val lp = model.logPerplexity(dataset)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")


    val topics = model.describeTopics(3)
    println("The topics described by their top-weighted terms:")
    topics.show(false)


    val transformed = model.transform(dataset)
    transformed.show(false)


    spark.stop()
  }
}
// scalastyle:on println
