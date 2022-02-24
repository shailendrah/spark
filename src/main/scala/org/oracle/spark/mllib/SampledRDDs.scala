// scalastyle:off println
package org.oracle.spark.mllib

import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils

/**
 * bin/run-example SampledRDDs
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object SampledRDDs {

  case class Params(input: String = "data/mllib/sample_binary_classification_data.txt")
    extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("SampledRDDs") {
      head("SampledRDDs: an example app for randomly generated and sampled RDDs.")
      opt[String]("input")
        .text(s"Input path to labeled examples in LIBSVM format, default: ${defaultParams.input}")
        .action((x, c) => c.copy(input = x))
      note(
        """
        |For example, the following command runs this app:
        |
        | bin/spark-submit --class SampledRDDs \
        |  examples/target/scala-*/spark-examples-*.jar
        """.stripMargin)
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = new SparkConf().setAppName(s"SampledRDDs with $params")
    val sc = new SparkContext(conf)

    val fraction = 0.1

    val examples = MLUtils.loadLibSVMFile(sc, params.input)
    val numExamples = examples.count()
    if (numExamples == 0) {
      throw new RuntimeException("Error: Data file had no samples to load.")
    }
    println(s"Loaded data with $numExamples examples from file: ${params.input}")


    val expectedSampleSize = (numExamples * fraction).toInt
    println(s"Sampling RDD using fraction $fraction.  Expected sample size = $expectedSampleSize.")
    val sampledRDD = examples.sample(withReplacement = true, fraction = fraction)
    println(s"  RDD.sample(): sample has ${sampledRDD.count()} examples")
    val sampledArray = examples.takeSample(withReplacement = true, num = expectedSampleSize)
    println(s"  RDD.takeSample(): sample has ${sampledArray.length} examples")

    println()


    val keyedRDD = examples.map { lp => (lp.label.toInt, lp.features) }
    println(s"  Keyed data using label (Int) as key ==> Orig")

    val keyCounts = keyedRDD.countByKey()


    val fractions = keyCounts.keys.map((_, fraction)).toMap
    val sampledByKeyRDD = keyedRDD.sampleByKey(withReplacement = true, fractions = fractions)
    val keyCountsB = sampledByKeyRDD.countByKey()
    val sizeB = keyCountsB.values.sum
    println(s"  Sampled $sizeB examples using approximate stratified sampling (by label)." +
      " ==> Approx Sample")


    val sampledByKeyRDDExact =
      keyedRDD.sampleByKeyExact(withReplacement = true, fractions = fractions)
    val keyCountsBExact = sampledByKeyRDDExact.countByKey()
    val sizeBExact = keyCountsBExact.values.sum
    println(s"  Sampled $sizeBExact examples using exact stratified sampling (by label)." +
      " ==> Exact Sample")


    println(s"   \tFractions of examples with key")
    println(s"Key\tOrig\tApprox Sample\tExact Sample")
    keyCounts.keys.toSeq.sorted.foreach { key =>
      val origFrac = keyCounts(key) / numExamples.toDouble
      val approxFrac = if (sizeB != 0) {
        keyCountsB.getOrElse(key, 0L) / sizeB.toDouble
      } else {
        0
      }
      val exactFrac = if (sizeBExact != 0) {
        keyCountsBExact.getOrElse(key, 0L) / sizeBExact.toDouble
      } else {
        0
      }
      println(s"$key\t$origFrac\t$approxFrac\t$exactFrac")
    }

    sc.stop()
  }
}
// scalastyle:on println
