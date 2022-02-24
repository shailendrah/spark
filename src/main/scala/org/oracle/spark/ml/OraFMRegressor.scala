// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.regression.{FMRegressionModel, FMRegressor}
// $example off$
import org.apache.spark.sql.SparkSession

object OraFMRegressor  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("FMRegressor")
      .getOrCreate()



    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")


    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .fit(data)


    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


    val fm = new FMRegressor()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setStepSize(0.001)


    val pipeline = new Pipeline()
      .setStages(Array(featureScaler, fm))


    val model = pipeline.fit(trainingData)


    val predictions = model.transform(testData)


    predictions.select("prediction", "label", "features").show(5)


    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val fmModel = model.stages(1).asInstanceOf[FMRegressionModel]
    println(s"Factors: ${fmModel.factors} Linear: ${fmModel.linear} " +
      s"Intercept: ${fmModel.intercept}")


    spark.stop()
  }
}
// scalastyle:on println
