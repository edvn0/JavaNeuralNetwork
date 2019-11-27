package math.activations;

import math.activations.ActivationFunction;
import matrix.Matrix;

public class ReluFunction implements ActivationFunction {

	private static final String NAME = "RELU";

	private double relu(double a) {
		return a > 0 ? a : 0;
	}

	private double reluDerivative(double a) {
		return a > 0 ? 1 : 0;
	}

	public ReluFunction() {}

	@Override
	public Matrix applyFunction(Matrix input, Matrix corr) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::relu);
		return returnMatrix;
	}

	@Override
	public Matrix applyDerivative(Matrix input, Matrix corr) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::reluDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("ReluFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
