// scalastyle:off println
package org.oracle.spark.mllib

import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib.util.MLUtils

/**
 * bin/run-example MultivariateSummarizer
 * By default, this loads a synthetic dataset from `data/mllib/sample_linear_regression_data.txt`.
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object MultivariateSummarizer {

  case class Params(input: String = "data/mllib/sample_linear_regression_data.txt")
    extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("MultivariateSummarizer") {
      head("MultivariateSummarizer: an example app for MultivariateOnlineSummarizer")
      opt[String]("input")
        .text(s"Input path to labeled examples in LIBSVM format, default: ${defaultParams.input}")
        .action((x, c) => c.copy(input = x))
      note(
        """
        |For example, the following command runs this app on a synthetic dataset:
        |
        | bin/spark-submit --class MultivariateSummarizer \
        |  examples/target/scala-*/spark-examples-*.jar \
        |  --input data/mllib/sample_linear_regression_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = new SparkConf().setAppName(s"MultivariateSummarizer with $params")
    val sc = new SparkContext(conf)

    val examples = MLUtils.loadLibSVMFile(sc, params.input).cache()

    println(s"Summary of data file: ${params.input}")
    println(s"${examples.count()} data points")


    val labelSummary = examples.aggregate(new MultivariateOnlineSummarizer())(
      (summary, lp) => summary.add(Vectors.dense(lp.label)),
      (sum1, sum2) => sum1.merge(sum2))


    val featureSummary = examples.aggregate(new MultivariateOnlineSummarizer())(
      (summary, lp) => summary.add(lp.features),
      (sum1, sum2) => sum1.merge(sum2))

    println()
    println(s"Summary statistics")
    println(s"\tLabel\tFeatures")
    println(s"mean\t${labelSummary.mean(0)}\t${featureSummary.mean.toArray.mkString("\t")}")
    println(s"var\t${labelSummary.variance(0)}\t${featureSummary.variance.toArray.mkString("\t")}")
    println(
      s"nnz\t${labelSummary.numNonzeros(0)}\t${featureSummary.numNonzeros.toArray.mkString("\t")}")
    println(s"max\t${labelSummary.max(0)}\t${featureSummary.max.toArray.mkString("\t")}")
    println(s"min\t${labelSummary.min(0)}\t${featureSummary.min.toArray.mkString("\t")}")
    println()

    sc.stop()
  }
}
// scalastyle:on println
