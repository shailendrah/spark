// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
// $example off$
import org.apache.spark.sql.SparkSession

object OraDecisionTreeRegression  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("DecisionTreeRegression")
      .getOrCreate()



    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")



    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)


    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")


    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))


    val model = pipeline.fit(trainingData)


    val predictions = model.transform(testData)


    predictions.select("prediction", "label", "features").show(5)


    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println(s"Learned regression tree model:\n ${treeModel.toDebugString}")


    spark.stop()
  }
}
// scalastyle:on println
