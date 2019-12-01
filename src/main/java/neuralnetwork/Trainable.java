package neuralnetwork;

import java.io.Serializable;
import math.activations.ActivationFunction;
import matrix.Matrix;
import org.ujmp.core.DenseMatrix;

public interface Trainable extends Serializable {

	/**
	 * Takes some values to classify, and compares to the correct classification. This method only
	 * takes 1 case,
	 *
	 * Uses a {@link ActivationFunction#applyFunction(Matrix)}
	 *
	 * Example usage: in: new Matrix[]{new Matrix(double[][]{0,1}), new Matrix(new double[][]{{1}}},
	 * method: "SGD" for the XOR problem.
	 *
	 * @param toFeedForward Matrix to feed forward.
	 * @param correct       labels for the data.
	 */
	void train(DenseMatrix toFeedForward, DenseMatrix correct);

	/**
	 * Predicts a classification from a double[] input.
	 *
	 * Uses a {@link ActivationFunction#applyDerivative(Matrix)}
	 *
	 * Example usage: double[]{0,1} becomes double[]{0.971} for the XOR problem.
	 *
	 * @param in values to be predicted.
	 *
	 * @return A classification of input values.
	 */
	DenseMatrix predict(DenseMatrix in);

	String toString();

}
