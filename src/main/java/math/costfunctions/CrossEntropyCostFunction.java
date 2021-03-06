package math.costfunctions;

import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

public class CrossEntropyCostFunction<M> implements CostFunction<M> {

	private static final String NAME = "Cross Entropy";

	@Override
	public double calculateCostFunction(final List<NetworkInput<M>> tData) {
		double size = tData.size();

		if (size == 1) {
			return calculateSingle(tData.get(0));
		}

		double loss = 0d;

		for (var data : tData) {
			loss += calculateSingle(data);
		}

		return -loss / size;
	}

	@Override
	public double calculateSingle(NetworkInput<M> data) {
		var inner = data.getLabel().hadamard(data.getData().add(1e-9).mapElements(Math::log));
		return inner.sum();
	}

	@Override
	public Matrix<M> applyCostFunctionGradient(final Matrix<M> input, final Matrix<M> correct) {
		return input.subtract(correct);
	}

	@Override
	public String name() {
		return NAME;
	}

}
