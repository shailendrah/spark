// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.clustering.PowerIterationClustering
// $example off$
import org.apache.spark.sql.SparkSession

object OraPowerIterationClustering  {
   def main(args: Array[String]): Unit = {
     val spark = SparkSession
       .builder
       .appName(s"${this.getClass.getSimpleName}")
       .getOrCreate()


     val dataset = spark.createDataFrame(Seq(
       (0L, 1L, 1.0),
       (0L, 2L, 1.0),
       (1L, 2L, 1.0),
       (3L, 4L, 1.0),
       (4L, 0L, 0.1)
     )).toDF("src", "dst", "weight")

     val model = new PowerIterationClustering().
       setK(2).
       setMaxIter(20).
       setInitMode("degree").
       setWeightCol("weight")

     val prediction = model.assignClusters(dataset).select("id", "cluster")


     prediction.show(false)


     spark.stop()
   }
 }
