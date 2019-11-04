package neuralnetwork;

import matrix.Matrix;

public interface Trainable {

	/**
	 * Takes some values to classify, and compares to the correct classification. This method only
	 * takes 1 case,
	 *
	 * Example usage: double[]{0,1} -> double[]{1} for the XOR problem.
	 *
	 * @param in array to classify
	 * @param correct array with the correct classification.
	 */
	void train(double[] in, double[] correct);

	/**
	 * Predicts a classification from a double[] input.
	 *
	 * Example usage: double[]{0,1} -> double[]{0.971} for the XOR problem.
	 *
	 * @param in values to be predicted.
	 * @return A classification of input values.
	 */
	Matrix predict(double[] in);

}
