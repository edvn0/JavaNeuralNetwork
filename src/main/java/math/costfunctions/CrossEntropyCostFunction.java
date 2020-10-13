package math.costfunctions;

import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

public class CrossEntropyCostFunction<M> implements CostFunction<M> {

	private static final String NAME = "Cross Entropy";

	@Override
	public double calculateCostFunction(final List<NetworkInput<M>> tData) {
		double size = tData.size();

		return -tData.parallelStream().map(e -> e.getData().mapElements(Math::log).hadamard(e.getLabel()))
				.peek(System.out::println).mapToDouble(Matrix::sum).sum() / size;

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
