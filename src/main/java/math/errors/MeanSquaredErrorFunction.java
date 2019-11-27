package math.errors;

import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public class MeanSquaredErrorFunction implements ErrorFunction {

	public MeanSquaredErrorFunction() {
	}

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		double sum = 0;

		for (NetworkInput networkInput : tData) {
			Matrix inner = networkInput.getLabel().subtract(networkInput.getData());
			inner = inner.map((e) -> e * e).map(e -> e / 2);
			sum += inner.getElement(0, 0);
		}

		sum /= tData.size();

		return sum;
	}

	@Override
	public Matrix applyErrorFunction(final Matrix input, final Matrix target) {
		return input.subtract(target);
	}

	@Override
	public Matrix applyErrorFunctionGradient(final Matrix in, final Matrix label) {
		return in.subtract(label);
	}

}
