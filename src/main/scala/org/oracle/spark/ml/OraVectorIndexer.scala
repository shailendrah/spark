// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.feature.VectorIndexer
// $example off$
import org.apache.spark.sql.SparkSession

object OraVectorIndexer  {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("VectorIndexer")
      .getOrCreate()


    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val indexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexed")
      .setMaxCategories(10)

    val indexerModel = indexer.fit(data)

    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} " +
      s"categorical features: ${categoricalFeatures.mkString(", ")}")


    val indexedData = indexerModel.transform(data)
    indexedData.show()


    spark.stop()
  }
}
// scalastyle:on println
