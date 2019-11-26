package math.activations;

import math.activations.ActivationFunction;
import matrix.Matrix;

public class TanhFunction implements ActivationFunction {

	private static final String NAME = "TANH";

	private double tanh(double a) {
		return Math.tanh(a);
	}

	private double tanhDerivative(double a) {
		return 1 - (a * a);
	}

	public TanhFunction() {

	}

	@Override
	public Matrix applyFunction(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::tanh);
		return returnMatrix;
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::tanhDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("TanhFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
