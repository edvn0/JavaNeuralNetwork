package math.activations;

import math.linearalgebra.Matrix;

public class DoNothingFunction<M> extends ActivationFunction<M> {

	public DoNothingFunction() {
	}

	@Override
	public Matrix<M> function(Matrix<M> m) {
		return m;
	}

	@Override
	public Matrix<M> derivative(Matrix<M> m) {
		return m.mapElements(e -> 1d);
	}

	@Override
	public void setValues(double in) {

	}

	@Override
	public String getName() {
		return "DoNothing";
	}

}
