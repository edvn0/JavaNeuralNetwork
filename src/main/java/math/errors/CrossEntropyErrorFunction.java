package math.errors;

import java.util.Arrays;
import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public class CrossEntropyErrorFunction implements ErrorFunction {

	private static double LOG_2_INV = 3.32192809489;

	public CrossEntropyErrorFunction() {

	}

	public double calculateCostFunction(final List<NetworkInput> tData) {
		double sum = 0;

		double[] stable = new double[tData.get(0).getData().getRows()];
		Arrays.fill(stable, 1e-6);

		Matrix stability = Matrix.fromArray(stable);

		for (NetworkInput input : tData) {
			Matrix label = input.getLabel();
			Matrix data = input.getData().add(stability);
			Matrix logged = data.map(this::log2);
			double dot = label.dotProduct(logged);
			sum += dot;
		}

		return (sum * -1) / tData.size();
	}

	private double log2(double e) {
		return Math.log(e) * LOG_2_INV;
	}

	@Override
	public Matrix applyErrorFunction(final Matrix in, final Matrix correct) {
		return null;
	}

	@Override
	public Matrix applyErrorFunctionGradient(final Matrix input, final Matrix label) {
		return input.subtract(label);
	}

}
