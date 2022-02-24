// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}

object OraStratifiedSampling {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("StratifiedSampling")
    val sc = new SparkContext(conf)



    val data = sc.parallelize(
      Seq((1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (2, 'e'), (3, 'f')))


    val fractions = Map(1 -> 0.1, 2 -> 0.6, 3 -> 0.3)


    val approxSample = data.sampleByKey(withReplacement = false, fractions = fractions)

    val exactSample = data.sampleByKeyExact(withReplacement = false, fractions = fractions)


    println(s"approxSample size is ${approxSample.collect().size}")
    approxSample.collect().foreach(println)

    println(s"exactSample its size is ${exactSample.collect().size}")
    exactSample.collect().foreach(println)

    sc.stop()
  }
}
// scalastyle:on println
