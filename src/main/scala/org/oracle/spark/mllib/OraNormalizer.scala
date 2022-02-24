// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// $example on$
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils
// $example off$

object OraNormalizer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Normalizer")
    val sc = new SparkContext(conf)


    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    val normalizer1 = new Normalizer()
    val normalizer2 = new Normalizer(p = Double.PositiveInfinity)


    val data1 = data.map(x => (x.label, normalizer1.transform(x.features)))


    val data2 = data.map(x => (x.label, normalizer2.transform(x.features)))


    println("data1: ")
    data1.collect.foreach(x => println(x))

    println("data2: ")
    data2.collect.foreach(x => println(x))

    sc.stop()
  }
}
// scalastyle:on println
