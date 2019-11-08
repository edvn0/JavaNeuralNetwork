package math;

import matrix.Matrix;

public class SigmoidFunction implements ActivationFunction {

	private static final String NAME = "SIGMOID";

	private double sigmoid(double in) {
		return 1 / (1 + Math.exp(-in));
	}

	private Double sigmoidDerivative(Double a) {
		return a * (1 - a);
	}

	@Override
	public Matrix functionToMatrix(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::sigmoid);
		return returnMatrix;
	}

	@Override
	public Matrix derivativeToMatrix(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::sigmoidDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}
}
