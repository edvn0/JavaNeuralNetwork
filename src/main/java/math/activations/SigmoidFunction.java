package math.activations;

import math.linearalgebra.Matrix;

public class SigmoidFunction<M> extends ActivationFunction<M> {


	@Override
	public Matrix<M> function(Matrix<M> m) {
		return m.mapElements(this::sigmoid);
	}

	private double sigmoid(double input) {
		return 1 / (1 + Math.exp(-input));
	}

	@Override
	public Matrix<M> derivative(Matrix<M> m) {
		return m.mapElements(this::sigmoidDerivative);
	}

	private double sigmoidDerivative(double input) {
		return input * (1 - input);
	}

	@Override
	public void setValues(double in) {
	}

	@Override
	public String getName() {
		return "Sigmoid";
	}
}
