// scalastyle:off println
package org.oracle.spark.ml

// $example on$
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataType, DataTypes}

// $example off$

/**
 * Create a custom [[org.apache.spark.ml.Transformer]] using
 * the [[OraUnaryTransformer]] abstraction.
 *
 * Run with
 * bin/run-example ml.OraUnaryTransformer
 */
object OraUnaryTransformer  {
  /**
   * Simple Transformer which adds a constant value to input Doubles.
   *
   * [[UnaryTransformer]] can be used to create a stage usable within Pipelines.
   * It defines parameters for specifying input and output columns:
   * [[UnaryTransformer.inputCol]] and [[UnaryTransformer.outputCol]].
   * It can optionally handle schema validation.
   *
   * [[DefaultParamsWritable]] provides a default implementation for persisting instances
   * of this Transformer.
   */
  class MyTransformer(override val uid: String)
    extends UnaryTransformer[Double, Double, MyTransformer] with DefaultParamsWritable {

    final val shift: DoubleParam = new DoubleParam(this, "shift", "Value added to input")

    def getShift: Double = $(shift)

    def setShift(value: Double): this.type = set(shift, value)

    def this() = this(Identifiable.randomUID("myT"))

    override protected def createTransformFunc: Double => Double = (input: Double) => {
      input + $(shift)
    }

    override protected def outputDataType: DataType = DataTypes.DoubleType

    override protected def validateInputType(inputType: DataType): Unit = {
      require(inputType == DataTypes.DoubleType, s"Bad input type: $inputType. Requires Double.")
    }
  }

  /**
   * Companion object Orafor our simple Transformer.
   *
   * [[DefaultParamsReadable]] provides a default implementation for loading instances
   * of this Transformer which were persisted using [[DefaultParamsWritable]].
   */
  object MyTransformer extends DefaultParamsReadable[MyTransformer]


  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("UnaryTransformer")
      .getOrCreate()


    val myTransformer = new MyTransformer()
      .setShift(0.5)
      .setInputCol("input")
      .setOutputCol("output")


    val data = spark.range(0, 5).toDF("input")
      .select(col("input").cast("double").as("input"))
    val result = myTransformer.transform(data)
    println("Transformed by adding constant value")
    result.show()


    val tmpDir = Utils.createTempDir()
    val dirName = tmpDir.getCanonicalPath
    myTransformer.write.overwrite().save(dirName)
    val sameTransformer = MyTransformer.load(dirName)


    println("Same transform applied from loaded model")
    val sameResult = sameTransformer.transform(data)
    sameResult.show()

    Utils.deleteRecursively(tmpDir)


    spark.stop()
  }
}
// scalastyle:on println

