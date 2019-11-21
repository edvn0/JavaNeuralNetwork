package math.activations;

import math.activations.ActivationFunction;
import matrix.Matrix;

public class SigmoidFunction implements ActivationFunction {

	private static final String NAME = "SIGMOID";

	private double sigmoid(double in) {
		return 1 / (1 + Math.exp(-in));
	}

	private double sigmoidDerivative(double a) {
		return a * (1 - a);
	}

	@Override
	public Matrix applyFunction(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::sigmoid);
		return returnMatrix;
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = returnMatrix.map(this::sigmoidDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("SigmoidFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
