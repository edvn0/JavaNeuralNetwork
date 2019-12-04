package math.errors;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;

public class BinaryCrossEntropyErrorFunction implements ErrorFunction {

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		double total = 0;
		for (NetworkInput s : tData) {
			double[][] yHat = s.getData().toDoubleArray();
			double[][] y = s.getLabel().toDoubleArray();

			double temp = 0;
			for (int i = 0; i < yHat.length; i++) {
				double label = y[i][0];
				double data = yHat[i][0];
				temp += label * log2(data) + (1 - label) * log2(1 - data);
			}
			total += temp / yHat.length;
		}
		return (total * -1) / tData.size();
	}

	private double log2(final double v) {
		return Math.log(v) / Math.log(2);
	}

	@Override
	public DenseMatrix applyErrorFunction(final DenseMatrix in, final DenseMatrix correct) {
		return (DenseMatrix) in.minus(correct);
	}

	@Override
	public DenseMatrix applyErrorFunctionGradient(final DenseMatrix in, final DenseMatrix label) {
		return (DenseMatrix) in.minus(label);
	}
}
