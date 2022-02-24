// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// $example on$
import org.apache.spark.mllib.feature.ElementwiseProduct
import org.apache.spark.mllib.linalg.Vectors
// $example off$

object OraElementwiseProduct {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("ElementwiseProduct")
    val sc = new SparkContext(conf)



    val data = sc.parallelize(Seq(Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0)))

    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct(transformingVector)


    val transformedData = transformer.transform(data)
    val transformedData2 = data.map(x => transformer.transform(x))


    println("transformedData: ")
    transformedData.collect.foreach(x => println(x))

    println("transformedData2: ")
    transformedData2.collect.foreach(x => println(x))

    sc.stop()
  }
}
// scalastyle:on println
