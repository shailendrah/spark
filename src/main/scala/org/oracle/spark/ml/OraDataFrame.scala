// scalastyle:off println
package org.oracle.spark.ml

import java.io.File
import scopt.OptionParser
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.util.Utils
import org.oracle.spark.mllib.AbstractParams

/**
 * ./bin/run-example ml.OraDataFrame  [options]
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object OraDataFrame  {

  case class Params(input: String = "data/mllib/sample_libsvm_data.txt")
    extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("DataFrame") {
      head("DataFrame: an example app using DataFrame for ML.")
      opt[String]("input")
        .text("input path to dataframe")
        .action((x, c) => c.copy(input = x))
      checkConfig { params =>
        success
      }
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"DataFrame  with $params")
      .getOrCreate()


    println(s"Loading LIBSVM file with UDT from ${params.input}.")
    val df: DataFrame = spark.read.format("libsvm").load(params.input).cache()
    println("Schema from LIBSVM:")
    df.printSchema()
    println(s"Loaded training data as a DataFrame with ${df.count()} records.")


    val labelSummary = df.describe("label")
    labelSummary.show()


    val features = df.select("features").rdd.map { case Row(v: Vector) => v }
    val featureSummary = features.aggregate(new MultivariateOnlineSummarizer())(
      (summary, feat) => summary.add(Vectors.fromML(feat)),
      (sum1, sum2) => sum1.merge(sum2))
    println(s"Selected features column with average values:\n ${featureSummary.mean.toString}")

    val tmpDir = Utils.createTempDir()
    val outputDir = new File(tmpDir, "dataframe").toString
    println(s"Saving to $outputDir as Parquet file.")
    df.write.parquet(outputDir)


    println(s"Loading Parquet file with UDT from $outputDir.")
    val newDF = spark.read.parquet(outputDir)
    println("Schema from Parquet:")
    newDF.printSchema()

    spark.stop()
  }
}
// scalastyle:on println
