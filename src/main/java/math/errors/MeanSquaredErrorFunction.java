package math.errors;

import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public class MeanSquaredErrorFunction implements ErrorFunction {

	public MeanSquaredErrorFunction() {
	}

	@Override
	public double calculateCostFunction(final List<NetworkInput> tData) {
		double[] sum = {0};
		tData.forEach((e) -> sum[0] += e.getLabel().subtract(e.getData()).magnitude() / 2);
		return sum[0];
	}

	@Override
	public Matrix applyErrorFunction(final Matrix input, final Matrix target) {
		return input.subtract(target);
	}

	@Override
	public Matrix applyErrorFunctionGradient(final Matrix in, final Matrix applied) {
		return null;
	}

}
