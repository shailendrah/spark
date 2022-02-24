// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * using Logistic Regression as the base classifier.
 * Run with
 * {{{
 * ./bin/run-example ml.OraOneVsRest
 * }}}
 */

object OraOneVsRest  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"OneVsRest")
      .getOrCreate()



    val inputData = spark.read.format("libsvm")
      .load("data/mllib/sample_multiclass_classification_data.txt")


    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))


    val classifier = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1E-6)
      .setFitIntercept(true)


    val ovr = new OneVsRest().setClassifier(classifier)


    val ovrModel = ovr.fit(train)


    val predictions = ovrModel.transform(test)


    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")


    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1 - accuracy}")


    spark.stop()
  }

}
// scalastyle:on println
