// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.test.ChiSqTestResult
import org.apache.spark.rdd.RDD
// $example off$

object OraHypothesisTesting {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("HypothesisTesting")
    val sc = new SparkContext(conf)



    val vec: Vector = Vectors.dense(0.1, 0.15, 0.2, 0.3, 0.25)



    val goodnessOfFitTestResult = Statistics.chiSqTest(vec)


    println(s"$goodnessOfFitTestResult\n")


    val mat: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))


    val independenceTestResult = Statistics.chiSqTest(mat)

    println(s"$independenceTestResult\n")

    val obs: RDD[LabeledPoint] =
      sc.parallelize(
        Seq(
          LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
          LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 0.0)),
          LabeledPoint(-1.0, Vectors.dense(-1.0, 0.0, -0.5)
          )
        )
      )




    val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)
    featureTestResults.zipWithIndex.foreach { case (k, v) =>
      println(s"Column ${(v + 1)} :")
      println(k)
    }


    sc.stop()
  }
}
// scalastyle:on println

