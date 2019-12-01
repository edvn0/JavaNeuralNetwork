package math.errors;

import java.util.Arrays;
import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class CrossEntropyErrorFunction implements ErrorFunction {

	private static double LOG_2_INV = 3.32192809489;

	public CrossEntropyErrorFunction() {

	}

	public double calculateCostFunction(final List<NetworkInput> tData) {
		double sum = 0;

		double[] stable = new double[(int) tData.get(0).getData().getRowCount()];
		Arrays.fill(stable, 1e-6);

		DenseMatrix stability = (DenseMatrix) Matrix.Factory.importFromArray(stable).transpose();

		for (NetworkInput input : tData) {
			double[][] label = input.getLabel().toDoubleArray();
			double[][] data = input.getData().plus(stability).toDoubleArray();

			double dot = 0;
			for (int i = 0; i < data.length; i++) {
				for (int j = 0; j < data[0].length; j++) {
					double logged = Math.log(data[i][j]) / Math.log(2);
					dot += label[i][j] * logged;
				}
			}

			sum += dot;
		}

		return (sum * -1) / tData.size();
	}

	private double log2(double e) {
		return Math.log(e) * LOG_2_INV;
	}

	@Override
	public DenseMatrix applyErrorFunction(final DenseMatrix in, final DenseMatrix correct) {
		return null;
	}

	@Override
	public DenseMatrix applyErrorFunctionGradient(final DenseMatrix input,
		final DenseMatrix label) {
		return (DenseMatrix) input.minus(label);
	}

}
