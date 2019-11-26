package math.activations;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import matrix.Matrix;

public class SoftmaxFunction implements ActivationFunction {

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

	@Override
	public Matrix applyDerivative(Matrix input) {
		return input.map((e) -> e * (1d - e));
	}

	public SoftmaxFunction() {

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
