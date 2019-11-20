package neuralnetwork;

import java.util.List;
import matrix.Matrix;

public interface Trainable {

	/**
	 * Takes some values to classify, and compares to the correct classification. This method only
	 * takes 1 case,
	 *
	 * Uses a {@link math.ActivationFunction#applyFunction(Matrix)}
	 *
	 * Example usage: in: new Matrix[]{new Matrix(double[][]{0,1}), new Matrix(new double[][]{{1}}},
	 * method: "SGD" for the XOR problem.
	 *
	 * @param training array of data and labels to classify. e.g. training.get(0)[0] = Matrix with
	 * input data and  training.get(0)[1] = Matrix with correct label.
	 * @param method what method is being used to train the network
	 */
	void train(List<Matrix[]> training, String method);

	/**
	 * Predicts a classification from a double[] input.
	 *
	 * Uses a {@link math.ActivationFunction#applyDerivative(Matrix)}
	 *
	 * Example usage: double[]{0,1} becomes double[]{0.971} for the XOR problem.
	 *
	 * @param in values to be predicted.
	 * @return A classification of input values.
	 */
	Matrix predict(Matrix in);

	String toString();

}
