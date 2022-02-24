// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.regression.{IsotonicRegression, IsotonicRegressionModel}
import org.apache.spark.mllib.util.MLUtils
// $example off$

object OraIsotonicRegression {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("IsotonicRegression")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,
      "data/mllib/sample_isotonic_regression_libsvm_data.txt").cache()


    val parsedData = data.map { labeledPoint =>
      (labeledPoint.label, labeledPoint.features(0), 1.0)
    }


    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)



    val model = new IsotonicRegression().setIsotonic(true).run(training)


    val predictionAndLabel = test.map { point =>
      val predictedLabel = model.predict(point._2)
      (predictedLabel, point._1)
    }


    val meanSquaredError = predictionAndLabel.map { case (p, l) => math.pow((p - l), 2) }.mean()
    println(s"Mean Squared Error = $meanSquaredError")


    model.save(sc, "target/tmp/myIsotonicRegressionModel")
    val sameModel = IsotonicRegressionModel.load(sc, "target/tmp/myIsotonicRegressionModel")


    sc.stop()
  }
}
// scalastyle:on println
