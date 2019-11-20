package utilities;

import java.util.stream.DoubleStream;
import matrix.Matrix;

public class MatrixUtilities {

	/**
	 * Return a probability distribution of the predicted values of {@link Matrix}.
	 *
	 * @param matrix {@link Matrix} with the predicted values
	 * @return SoftMax(exponential probability distribution) of the prediction.
	 */
	public static Matrix networkOutputsSoftMax(Matrix matrix) {
		Matrix copy = matrix;
		double softMaxTotal = DoubleStream.of(copy.transpose().getData()[0])
			.map(Math::exp).sum();
		Matrix transposed = copy.transpose();

		for (int i = 0; i < transposed.getData().length; i++) {
			double value = Math.exp(transposed.getData()[i][0]);
			transposed.getData()[i][0] = value / softMaxTotal;
		}

		return transposed.transpose();
	}

	/**
	 * Output the index of the max element
	 *
	 * @param input Output data of the prediction network
	 * @return index of max element
	 */
	public static int networkOutputsMax(Matrix input) {
		input = networkOutputsSoftMax(input);
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
