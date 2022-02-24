// scalastyle:off println
package org.oracle.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/**
 * Compute the singular value decomposition (SVD) of a tall-and-skinny matrix.
 *
 * The input matrix must be stored in row-oriented dense format, one line per row with its entries
 * separated by space. For example,
 * {{{
 * 0.5 1.0
 * 2.0 3.0
 * 4.0 5.0
 * }}}
 * represents a 3-by-2 matrix, whose first row is (0.5, 1.0).
 */
object TallSkinnySVD {
  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      System.err.println("Usage: TallSkinnySVD <input>")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("TallSkinnySVD")
    val sc = new SparkContext(conf)


    val rows = sc.textFile(args(0)).map { line =>
      val values = line.split(' ').map(_.toDouble)
      Vectors.dense(values)
    }
    val mat = new RowMatrix(rows)


    val svd = mat.computeSVD(mat.numCols().toInt)

    println(s"Singular values are ${svd.s}")

    sc.stop()
  }
}
// scalastyle:on println
