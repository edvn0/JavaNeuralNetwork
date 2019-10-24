package math;

import matrix.Matrix;

public class TanhFunction implements ActivationFunction {

	private static final String NAME = "TANH";

	private double tanh(double a) {
		return Math.tanh(a);
	}

	private double tanhDerivative(double a) {
		return 1 - (a * a);
	}

	@Override
	public Matrix applyFunctionToMatrix(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::tanh);
		return returnMatrix;
	}

	@Override
	public Matrix applyDerivativeFunctionToMatrix(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::tanhDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return null;
	}
}
