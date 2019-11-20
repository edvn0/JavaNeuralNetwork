package math;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import matrix.Matrix;

public class SoftmaxFunction implements ActivationFunction {

	private Matrix softMax(Matrix input) {
		if (input.getColumns() != 1) {
			throw new IllegalArgumentException("You can only perform SoftMax on a vector.");
		}

		double[] data = new double[input.getRows()];
		for (int i = 0; i < input.getData().length; i++) {
			data[i] = input.getData()[i][0];
		}

		double max = this.max(data);
		double sum = DoubleStream.of(data).map(Math::exp).sum();
		double[] outputData = new double[data.length];

		for (int i = 0; i < outputData.length; i++) {
			outputData[i] = (Math.exp(data[i] - max) / sum);
		}

		System.out.println(Arrays.toString(outputData));

		return Matrix.fromArray(outputData);
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
		return input.map((e) -> e * (1d - e));
	}

	@Override
	public String getName() {
		return "SOFTMAX";
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("SoftmaxFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
