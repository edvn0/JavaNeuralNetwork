package utilities;

import matrix.Matrix;

public class MatrixUtilities {

	/**
	 * Return a probability distribution of the predicted values of {@link Matrix}.
	 *
	 * @param matrix {@link Matrix} with the predicted values
	 *
	 * @return SoftMax(exponential probability distribution) of the prediction.
	 */
	public static Matrix networkOutputsSoftMax(Matrix matrix) {
		Matrix copy = matrix;

		Matrix mapped = matrix.map(Math::exp);
		double softMaxTotal = mapped.matrixSum();
		System.out.println(softMaxTotal);

		double[] newMatrix = new double[copy.getRows()];

		for (int i = 0; i < matrix.getRows(); i++) {
			double inMatrix = matrix.getElement(i, 0);
			System.out.println("inMatrix: " + inMatrix);
			double value = Math.exp(matrix.getElement(i, 0));
			System.out.println("value: " + value);
			newMatrix[i] = value / softMaxTotal;
			System.out.println("matrix value: " + newMatrix[i]);
		}

		return Matrix.fromArray(newMatrix);
	}

	/**
	 * Output the index of the max element
	 *
	 * @param input Output data of the prediction network
	 *
	 * @return index of max element
	 */
	public static int networkOutputsMax(Matrix input) {
		double max = input.getData()[0][0];
		int index = 0;
		for (int i = 0; i < input.getData().length; i++) {
			double[][] data = input.getData();
			if (data[i][0] > max) {
				max = data[i][0];
				index = i;
			}
		}
		return index;
	}
}
