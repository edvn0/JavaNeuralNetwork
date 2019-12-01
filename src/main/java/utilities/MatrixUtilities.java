package utilities;

import java.util.function.UnaryOperator;
import matrix.Matrix;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class MatrixUtilities {

	public static double[] toArray(DenseMatrix b) {
		return b.transpose().toDoubleArray()[0];
	}

	public static DenseMatrix map(DenseMatrix in, UnaryOperator<Double> t) {
		double[][] values = in.toDoubleArray();
		double[][] out = new double[values.length][values[0].length];
		for (int i = 0; i < values.length; i++) {
			for (int j = 0; j < values[0].length; j++) {
				double temp = values[i][j];
				double applied = t.apply(temp);
				out[i][j] = applied;
			}
		}
		return org.ujmp.core.Matrix.Factory.importFromArray(out);
	}

	/**
	 * Return a probability distribution of the predicted values of {@link Matrix}.
	 *
	 * @param matrix {@link Matrix} with the predicted values
	 *
	 * @return SoftMax(exponential probability distribution) of the prediction.
	 */
	public static Matrix networkOutputsSoftMax(Matrix matrix) {

		Matrix mapped = matrix.map(Math::exp);
		double softMaxTotal = mapped.matrixSum();
		System.out.println(softMaxTotal);

		double[] newMatrix = new double[matrix.getRows()];

		for (int i = 0; i < matrix.getRows(); i++) {
			double inValue = matrix.getElement(i, 0);
			double value = Math.exp(inValue);
			newMatrix[i] = value / softMaxTotal;
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
	public static int argMax(Matrix input) {
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

	public static int argMax(DenseMatrix input) {
		org.ujmp.core.Matrix argMax = input.indexOfMax(Ret.NEW, 0);
		return argMax.intValue();

	}

	public static void isNan(final DenseMatrix x) {
		double[][] data = x.toDoubleArray();
		for (double[] ds : data) {
			for (double d : ds) {
				if (Double.isNaN(d)) {
					throw new RuntimeException("Matrix is NaN");
				}
			}
		}
	}
}
