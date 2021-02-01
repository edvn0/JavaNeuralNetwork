package math.activations;

import math.linearalgebra.Matrix;

public class TanhFunction<M> extends ActivationFunction<M> {

	@Override
	public Matrix<M> function(Matrix<M> m) {
		return m.mapElements(this::tanh);
	}

	private double tanh(double a) {
		return Math.tanh(a);
	}

	@Override
	public Matrix<M> derivative(Matrix<M> m) {
		return m.mapElements(this::tanhDerivative);
	}

	private double tanhDerivative(double a) {
		return 1 - (a * a);
	}

	@Override
	public void setValues(double in) {

	}

	@Override
	public String getName() {
		return "Tanh";
	}
}
