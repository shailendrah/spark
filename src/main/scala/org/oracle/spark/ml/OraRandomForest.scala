// scalastyle:off println
package org.oracle.spark.ml

import java.util.Locale
import scopt.OptionParser
import scala.collection.mutable
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.oracle.spark.mllib.AbstractParams


/**
 * ./bin/run-example ml.OraRandomForest  [options]
 * Decision Trees and ensembles can take a large amount of memory. If the run-example command
 * above fails, try running via spark-submit and specifying the amount of memory as at least 1g.
 * For local mode, run
 * ./bin/spark-submit --class RandomForest  --driver-memory 1g
 *   [examples JAR path] [options]
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object OraRandomForest  {

  case class Params(
      input: String = null,
      testInput: String = "",
      dataFormat: String = "libsvm",
      algo: String = "classification",
      maxDepth: Int = 5,
      maxBins: Int = 32,
      minInstancesPerNode: Int = 1,
      minInfoGain: Double = 0.0,
      numTrees: Int = 10,
      featureSubsetStrategy: String = "auto",
      fracTest: Double = 0.2,
      cacheNodeIds: Boolean = false,
      checkpointDir: Option[String] = None,
      checkpointInterval: Int = 10) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("RandomForest") {
      head("RandomForest: an example random forest app.")
      opt[String]("algo")
        .text(s"algorithm (classification, regression), default: ${defaultParams.algo}")
        .action((x, c) => c.copy(algo = x))
      opt[Int]("maxDepth")
        .text(s"max depth of the tree, default: ${defaultParams.maxDepth}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("maxBins")
        .text(s"max number of bins, default: ${defaultParams.maxBins}")
        .action((x, c) => c.copy(maxBins = x))
      opt[Int]("minInstancesPerNode")
        .text(s"min number of instances required at child nodes to create the parent split," +
        s" default: ${defaultParams.minInstancesPerNode}")
        .action((x, c) => c.copy(minInstancesPerNode = x))
      opt[Double]("minInfoGain")
        .text(s"min info gain required to create a split, default: ${defaultParams.minInfoGain}")
        .action((x, c) => c.copy(minInfoGain = x))
      opt[Int]("numTrees")
        .text(s"number of trees in ensemble, default: ${defaultParams.numTrees}")
        .action((x, c) => c.copy(numTrees = x))
      opt[String]("featureSubsetStrategy")
        .text(s"number of features to use per node (supported:" +
        s" ${RandomForestClassifier.supportedFeatureSubsetStrategies.mkString(",")})," +
        s" default: ${defaultParams.numTrees}")
        .action((x, c) => c.copy(featureSubsetStrategy = x))
      opt[Double]("fracTest")
        .text(s"fraction of data to hold out for testing. If given option testInput, " +
        s"this option is ignored. default: ${defaultParams.fracTest}")
        .action((x, c) => c.copy(fracTest = x))
      opt[Boolean]("cacheNodeIds")
        .text(s"whether to use node Id cache during training, " +
        s"default: ${defaultParams.cacheNodeIds}")
        .action((x, c) => c.copy(cacheNodeIds = x))
      opt[String]("checkpointDir")
        .text(s"checkpoint directory where intermediate node Id caches will be stored, " +
        s"default: ${
          defaultParams.checkpointDir match {
            case Some(strVal) => strVal
            case None => "None"
          }
        }")
        .action((x, c) => c.copy(checkpointDir = Some(x)))
      opt[Int]("checkpointInterval")
        .text(s"how often to checkpoint the node Id cache, " +
        s"default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))
      opt[String]("testInput")
        .text(s"input path to test dataset. If given, option fracTest is ignored." +
        s" default: ${defaultParams.testInput}")
        .action((x, c) => c.copy(testInput = x))
      opt[String]("dataFormat")
        .text("data format: libsvm (default), dense (deprecated in Spark v1.1)")
        .action((x, c) => c.copy(dataFormat = x))
      arg[String]("<input>")
        .text("input path to labeled examples")
        .required()
        .action((x, c) => c.copy(input = x))
      checkConfig { params =>
        if (params.fracTest < 0 || params.fracTest >= 1) {
          failure(s"fracTest ${params.fracTest} value incorrect; should be in [0,1).")
        } else {
          success
        }
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
      .appName(s"RandomForest  with $params")
      .getOrCreate()

    params.checkpointDir.foreach(spark.sparkContext.setCheckpointDir)
    val algo = params.algo.toLowerCase(Locale.ROOT)

    println(s"RandomForest  with parameters:\n$params")


    val (training: DataFrame, test: DataFrame) = OraDecisionTree.loadDatasets(params.input,
      params.dataFormat, params.testInput, algo, params.fracTest)


    val stages = new mutable.ArrayBuffer[PipelineStage]()

    val labelColName = if (algo == "classification") "indexedLabel" else "label"
    if (algo == "classification") {
      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol(labelColName)
      stages += labelIndexer
    }


    val featuresIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(10)
    stages += featuresIndexer

    val dt = algo match {
      case "classification" =>
        new RandomForestClassifier()
          .setFeaturesCol("indexedFeatures")
          .setLabelCol(labelColName)
          .setMaxDepth(params.maxDepth)
          .setMaxBins(params.maxBins)
          .setMinInstancesPerNode(params.minInstancesPerNode)
          .setMinInfoGain(params.minInfoGain)
          .setCacheNodeIds(params.cacheNodeIds)
          .setCheckpointInterval(params.checkpointInterval)
          .setFeatureSubsetStrategy(params.featureSubsetStrategy)
          .setNumTrees(params.numTrees)
      case "regression" =>
        new RandomForestRegressor()
          .setFeaturesCol("indexedFeatures")
          .setLabelCol(labelColName)
          .setMaxDepth(params.maxDepth)
          .setMaxBins(params.maxBins)
          .setMinInstancesPerNode(params.minInstancesPerNode)
          .setMinInfoGain(params.minInfoGain)
          .setCacheNodeIds(params.cacheNodeIds)
          .setCheckpointInterval(params.checkpointInterval)
          .setFeatureSubsetStrategy(params.featureSubsetStrategy)
          .setNumTrees(params.numTrees)
      case _ => throw new IllegalArgumentException(s"Algo ${params.algo} not supported.")
    }
    stages += dt
    val pipeline = new Pipeline().setStages(stages.toArray)


    val startTime = System.nanoTime()
    val pipelineModel = pipeline.fit(training)
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $elapsedTime seconds")


    algo match {
      case "classification" =>
        val rfModel = pipelineModel.stages.last.asInstanceOf[RandomForestClassificationModel]
        if (rfModel.totalNumNodes < 30) {
          println(rfModel.toDebugString)
        } else {
          println(rfModel)
        }
      case "regression" =>
        val rfModel = pipelineModel.stages.last.asInstanceOf[RandomForestRegressionModel]
        if (rfModel.totalNumNodes < 30) {
          println(rfModel.toDebugString)
        } else {
          println(rfModel)
        }
      case _ => throw new IllegalArgumentException(s"Algo ${params.algo} not supported.")
    }


    algo match {
      case "classification" =>
        println("Training data results:")
        OraDecisionTree.evaluateClassificationModel(pipelineModel, training, labelColName)
        println("Test data results:")
        OraDecisionTree.evaluateClassificationModel(pipelineModel, test, labelColName)
      case "regression" =>
        println("Training data results:")
        OraDecisionTree.evaluateRegressionModel(pipelineModel, training, labelColName)
        println("Test data results:")
        OraDecisionTree.evaluateRegressionModel(pipelineModel, test, labelColName)
      case _ =>
        throw new IllegalArgumentException(s"Algo ${params.algo} not supported.")
    }

    spark.stop()
  }
}
// scalastyle:on println
