package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.fpm.FPGrowth
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * Run with
 * {{{
 * bin/run-example ml.OraFPGrowth
 * }}}
 */
object OraFPGrowth  {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()
    import spark.implicits._


    val dataset = spark.createDataset(Seq(
      "1 2 5",
      "1 2 3 5",
      "1 2")
    ).map(t => t.split(" ")).toDF("items")

    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.5).setMinConfidence(0.6)
    val model = fpgrowth.fit(dataset)


    model.freqItemsets.show()


    model.associationRules.show()



    model.transform(dataset).show()


    spark.stop()
  }
}
