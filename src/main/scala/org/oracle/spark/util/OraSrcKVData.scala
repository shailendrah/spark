package org.oracle.spark.util

import oracle.kv.hadoop.table.TableInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext

import java.util.Comparator
import java.util.stream.Collectors
import org.apache.spark.api.java.JavaRDD
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object OraSrcKVData {

  val STORE_NAME = "kvstore"
  val TABLE_NAME = "Log"
  val KVHOST = "localhost:5000"

  def main(args: Array[String]): Unit = {
    if (args.length % 2 != 0) {
      System.exit(0)
    }
    if (args.length == 1 && "--help" == args(0)) {
      System.exit(0)
    }

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("Remoting").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("ALS")
      .getOrCreate()
    val df = spark.read.format("io.oranosql.v3io.spark.sql.kv")
      .load("v3io://mycontainer/src/table")

    val hconf = new Configuration
    hconf.set("oracle.kv.kvstore", STORE_NAME)
    hconf.set("oracle.kv.tableName", TABLE_NAME)
    hconf.set("oracle.kv.hosts", KVHOST)

    /*
    df.write.format("io.oranosql.v3io.spark.sql.kv")
      .option("key", "id")
      .save("v3io://mycontainer/dest_table")
     */

  }
  
}
