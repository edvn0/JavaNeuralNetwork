package math;

import matrix.Matrix;

public class LinearFunction implements ActivationFunction {

	public double linear(double a) {
		return a;
	}

	public double linearDerivative(double a) {
		return 1;
	}


	@Override
	public Matrix applyFunction(Matrix input) {
		return input.map(this::linear);
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		return input.map(this::linearDerivative);
	}

	@Override
	public String getName() {
		return "LIN";
	}
}
