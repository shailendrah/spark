// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
// $example off$

object OraSVMWithSGD {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SVMWithSGD")
    val sc = new SparkContext(conf)



    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")


    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)


    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)


    model.clearThreshold()


    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }


    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println(s"Area under ROC = $auROC")


    model.save(sc, "target/tmp/scalaSVMWithSGDModel")
    val sameModel = SVMModel.load(sc, "target/tmp/scalaSVMWithSGDModel")


    sc.stop()
  }
}
// scalastyle:on println
