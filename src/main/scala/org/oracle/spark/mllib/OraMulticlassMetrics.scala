// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
// $example off$

object OraMulticlassMetrics {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("MulticlassMetrics")
    val sc = new SparkContext(conf)



    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_multiclass_classification_data.txt")


    val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    training.cache()


    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(3)
      .run(training)


    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }


    val metrics = new MulticlassMetrics(predictionAndLabels)


    println("Confusion matrix:")
    println(metrics.confusionMatrix)


    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")


    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }


    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }


    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }


    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }


    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


    sc.stop()
  }
}
// scalastyle:on println
