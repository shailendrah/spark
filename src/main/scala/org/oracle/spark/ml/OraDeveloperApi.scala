// scalastyle:off println
package org.oracle.spark.ml

import org.apache.spark.ml.classification.{ClassificationModel, Classifier, ClassifierParams}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, Row, SparkSession}

/**
 * A simple example demonstrating how to write your own learning algorithm using Estimator,
 * Transformer, and other abstractions.
 * This mimics [[org.apache.spark.ml.classification.LogisticRegression]].
 * Run with
 * {{{
 * bin/run-example ml.DeveloperApi
 * }}}
 */
object OraDeveloperApi  {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("DeveloperApi")
      .getOrCreate()


    val training = spark.createDataFrame(Seq(
      LabeledPoint(1.0, Vectors.dense(0.0, 1.1, 0.1)),
      LabeledPoint(0.0, Vectors.dense(2.0, 1.0, -1.0)),
      LabeledPoint(0.0, Vectors.dense(2.0, 1.3, 1.0)),
      LabeledPoint(1.0, Vectors.dense(0.0, 1.2, -0.5))))


    val lr = new MyLogisticRegression()

    println(s"MyLogisticRegression parameters:\n ${lr.explainParams()}")


    lr.setMaxIter(10)


    val model = lr.fit(training.toDF())


    val test = spark.createDataFrame(Seq(
      LabeledPoint(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      LabeledPoint(0.0, Vectors.dense(3.0, 2.0, -0.1)),
      LabeledPoint(1.0, Vectors.dense(0.0, 2.2, -1.5))))


    val sumPredictions: Double = model.transform(test)
      .select("features", "label", "prediction")
      .collect()
      .map { case Row(features: Vector, label: Double, prediction: Double) =>
        prediction
      }.sum
    assert(sumPredictions == 0.0,
      "MyLogisticRegression predicted something other than 0, even though all coefficients are 0!")

    spark.stop()
  }
}

/**
 *   of defining a parameter trait for a user-defined type of [[Classifier]].
 *
 * NOTE: This is private since it is an example. In practice, you may not want it to be private.
 */
private trait MyLogisticRegressionParams extends ClassifierParams {

  /**
   * Param for max number of iterations
   *
   * NOTE: The usual way to add a parameter to a model or algorithm is to include:
   *   - val myParamName: ParamType
   *   - def getMyParamName
   *   - def setMyParamName
   * Here, we have a trait to be mixed in with the Estimator and Model (MyLogisticRegression
   * and MyLogisticRegressionModel). We place the setter (setMaxIter) method in the Estimator
   * class since the maxIter parameter is only used during training (not in the Model).
   */
  val maxIter: IntParam = new IntParam(this, "maxIter", "max number of iterations")
  def getMaxIter: Int = $(maxIter)
}

/**
 *   of defining a type of [[Classifier]].
 *
 * NOTE: This is private since it is an example. In practice, you may not want it to be private.
 */
private class MyLogisticRegression(override val uid: String)
  extends Classifier[Vector, MyLogisticRegression, MyLogisticRegressionModel]
  with MyLogisticRegressionParams {

  def this() = this(Identifiable.randomUID("myLogReg"))

  setMaxIter(100)


  def setMaxIter(value: Int): this.type = set(maxIter, value)


  override protected def train(dataset: Dataset[_]): MyLogisticRegressionModel = {

    val oldDataset = extractLabeledPoints(dataset)


    val numFeatures = oldDataset.take(1)(0).features.size
    val coefficients = Vectors.zeros(numFeatures)


    new MyLogisticRegressionModel(uid, coefficients).setParent(this)
  }

  override def copy(extra: ParamMap): MyLogisticRegression = defaultCopy(extra)
}

/**
 *   of defining a type of [[ClassificationModel]].
 *
 * NOTE: This is private since it is an example. In practice, you may not want it to be private.
 */
private class MyLogisticRegressionModel(
    override val uid: String,
    val coefficients: Vector)
  extends ClassificationModel[Vector, MyLogisticRegressionModel]
  with MyLogisticRegressionParams {

  /**
   * Raw prediction for each possible label.
   * The meaning of a "raw" prediction may vary between algorithms, but it intuitively gives
   * a measure of confidence in each possible label (where larger = more confident).
   * This internal method is used to implement [[transform()]] and output [[rawPredictionCol]].
   *
   * @return  vector where element i is the raw prediction for label i.
   *          This raw prediction may be any real number, where a larger value indicates greater
   *          confidence for that label.
   */
  override def predictRaw(features: Vector): Vector = {
    val margin = BLAS.dot(features, coefficients)


    Vectors.dense(-margin, margin)
  }


  override val numClasses: Int = 2


  override val numFeatures: Int = coefficients.size

  /**
   * Create a copy of the model.
   * The copy is shallow, except for the embedded paramMap, which gets a deep copy.
   *
   * This is used for the default implementation of [[transform()]].
   */
  override def copy(extra: ParamMap): MyLogisticRegressionModel = {
    copyValues(new MyLogisticRegressionModel(uid, coefficients), extra).setParent(parent)
  }
}
// scalastyle:on println
