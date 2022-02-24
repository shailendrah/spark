// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{FMClassificationModel, FMClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StringIndexer}
// $example off$
import org.apache.spark.sql.SparkSession

object OraFMClassifier  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("FMClassifier")
      .getOrCreate()



    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")



    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .fit(data)


    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


    val fm = new FMClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("scaledFeatures")
      .setStepSize(0.001)


    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))


    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureScaler, fm, labelConverter))


    val model = pipeline.fit(trainingData)


    val predictions = model.transform(testData)


    predictions.select("predictedLabel", "label", "features").show(5)


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

    val fmModel = model.stages(2).asInstanceOf[FMClassificationModel]
    println(s"Factors: ${fmModel.factors} Linear: ${fmModel.linear} " +
      s"Intercept: ${fmModel.intercept}")


    spark.stop()
  }
}
// scalastyle:on println
