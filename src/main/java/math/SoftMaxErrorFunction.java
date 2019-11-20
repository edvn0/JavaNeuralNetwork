package math;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import matrix.Matrix;

public class SoftMaxErrorFunction implements ActivationFunction, ErrorFunction {

	private Matrix softMax(Matrix input) {
		if (input.getColumns() != 1) {
			throw new IllegalArgumentException("You can only perform SoftMax on a vector.");
		}

		Matrix max = this.max(input);
		Matrix z = input.subtract(max);
		double sum = DoubleStream.of(z.toArray()).map(Math::exp).sum();

		return z.map((e) -> Math.exp(e) / sum);
	}

	private Matrix max(Matrix input) {
		double[] inputs = input.toArray();
		double max = this.max(inputs);
		Arrays.fill(inputs, max);
		return Matrix.fromArray(inputs);
	}

	private double max(double[] data) {
		double max = data[0];
		for (int i = 1; i < data.length; i++) {
			if (data[i] > max) {
				max = data[i];
			}
		}
		return max;
	}

	@Override
	public Matrix applyFunction(Matrix input) {
		return this.softMax(input);
	}

	/**
	 * Will not be used, returns null.
	 */
	@Override
	public Matrix applyDerivative(Matrix input) {
		for (int i = 0; i < input.getRows(); i++) {

		}
		return null;
	}

	private int kroneckerDelta(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	@Override
	public String getName() {
		return null;
	}

	/**
	 * Returns the derivative of the error function.
	 */
	@Override
	public Matrix applyErrorFunction(Matrix input, Matrix target) {
		Matrix errors = input.subtract(target);

		int size = errors.getRows();

		double[] deltaAj = new double[size];
		for (int j = 0; j < size; j++) {
			double sum = 0;
			for (int i = 0; i < size; i++) {
				double x1 = kroneckerDelta(j, i) - input.getElement(j, 0);
				double x2 = input.getElement(i, 0);
				sum += errors.getElement(i, 0) * x1 * x2;
			}
			deltaAj[j] = sum;
		}
		return Matrix.fromArray(deltaAj);
	}
}
