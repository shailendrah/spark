// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
// $example off$

object OraBinaryClassificationMetrics {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("BinaryClassificationMetrics")
    val sc = new SparkContext(conf)


    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_binary_classification_data.txt")


    val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    training.cache()


    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)


    model.clearThreshold


    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }


    val metrics = new BinaryClassificationMetrics(predictionAndLabels)


    val precision = metrics.precisionByThreshold
    precision.collect.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }


    val recall = metrics.recallByThreshold
    recall.collect.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }


    val PRC = metrics.pr


    val f1Score = metrics.fMeasureByThreshold
    f1Score.collect.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    fScore.collect.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }


    val auPRC = metrics.areaUnderPR
    println(s"Area under precision-recall curve = $auPRC")


    val thresholds = precision.map(_._1)


    val roc = metrics.roc


    val auROC = metrics.areaUnderROC
    println(s"Area under ROC = $auROC")

    sc.stop()
  }
}
// scalastyle:on println
