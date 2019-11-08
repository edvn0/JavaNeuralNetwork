package math;

import matrix.Matrix;

public class ReluFunction implements ActivationFunction {

	private static final String NAME = "RELU";

	private double relu(double a) {
		return a > 0 ? a : 0;
	}

	private double reluDerivative(double a) {
		return a > 0 ? 1 : 0;
	}

	@Override
	public Matrix functionToMatrix(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::relu);
		return returnMatrix;
	}

	@Override
	public Matrix derivativeToMatrix(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::reluDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}
}
