// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.util.MLUtils

/**
 * ./bin/run-example SparseNaiveBayes [options] <input>
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object SparseNaiveBayes {

  case class Params(
      input: String = null,
      minPartitions: Int = 0,
      numFeatures: Int = -1,
      lambda: Double = 1.0) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("SparseNaiveBayes") {
      head("SparseNaiveBayes: an example naive Bayes app for LIBSVM data.")
      opt[Int]("numPartitions")
        .text("min number of partitions")
        .action((x, c) => c.copy(minPartitions = x))
      opt[Int]("numFeatures")
        .text("number of features")
        .action((x, c) => c.copy(numFeatures = x))
      opt[Double]("lambda")
        .text(s"lambda (smoothing constant), default: ${defaultParams.lambda}")
        .action((x, c) => c.copy(lambda = x))
      arg[String]("<input>")
        .text("input paths to labeled examples in LIBSVM format")
        .required()
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = new SparkConf().setAppName(s"SparseNaiveBayes with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val minPartitions =
      if (params.minPartitions > 0) params.minPartitions else sc.defaultMinPartitions

    val examples =
      MLUtils.loadLibSVMFile(sc, params.input, params.numFeatures, minPartitions)

    examples.cache()

    val splits = examples.randomSplit(Array(0.8, 0.2))
    val training = splits(0)
    val test = splits(1)

    val numTraining = training.count()
    val numTest = test.count()

    println(s"numTraining = $numTraining, numTest = $numTest.")

    val model = new NaiveBayes().setLambda(params.lambda).run(training)

    val prediction = model.predict(test.map(_.features))
    val predictionAndLabel = prediction.zip(test.map(_.label))
    val accuracy = predictionAndLabel.filter(x => x._1 == x._2).count().toDouble / numTest

    println(s"Test accuracy = $accuracy.")

    sc.stop()
  }
}
// scalastyle:on println
