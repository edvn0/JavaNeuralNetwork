package math.activations;

import math.linearalgebra.Matrix;

public class LinearFunction<M> extends ActivationFunction<M> {

	private double value;

	public LinearFunction(double value) {
		this.value = value;
	}

	/**
	 * Default constructor sets value to 1.
	 */
	public LinearFunction() {
		this.value = 1;
	}

	@Override
	public Matrix<M> function(Matrix<M> m) {
		return m.mapElements((e) -> e * value);
	}

	@Override
	public Matrix<M> derivative(Matrix<M> m) {
		return m.mapElements((e) -> value);
	}

	@Override
	public void setValues(double in) {
		this.value = in;
	}

	@Override
	public String getName() {
		return "Linear";
	}
}
