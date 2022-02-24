// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// $example on$
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
// $example off$

object OraStandardScaler {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("StandardScaler")
    val sc = new SparkContext(conf)


    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    val scaler1 = new StandardScaler().fit(data.map(x => x.features))
    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))

    val scaler3 = new StandardScalerModel(scaler2.std, scaler2.mean)


    val data1 = data.map(x => (x.label, scaler1.transform(x.features)))


    val data2 = data.map(x => (x.label, scaler2.transform(Vectors.dense(x.features.toArray))))


    println("data1: ")
    data1.collect.foreach(x => println(x))

    println("data2: ")
    data2.collect.foreach(x => println(x))

    sc.stop()
  }
}
// scalastyle:on println
