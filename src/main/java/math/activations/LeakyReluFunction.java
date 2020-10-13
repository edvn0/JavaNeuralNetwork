package math.activations;

import math.linearalgebra.Matrix;

public class LeakyReluFunction<M> extends ReluFunction<M> {

	private double alpha;

	public LeakyReluFunction(double alpha) {
		super();
		this.alpha = alpha;
	}

	public LeakyReluFunction() {
		this.alpha = 0.01;
	}

	@Override
	public void setValues(double in) {
		this.alpha = in;
	}

	@Override
	public String getName() {
		return "LeakyReLU";
	}

	@Override
	public Matrix<M> function(Matrix<M> in) {
		return in.mapElements((e) -> e > 0 ? e : alpha * e);
	}

	@Override
	public Matrix<M> derivative(Matrix<M> in) {
		return in.mapElements((e) -> e > 0 ? 1 : alpha);
	}

}
