package math.errors;

import matrix.Matrix;

public class CrossEntropyErrorFunction implements ErrorFunction {

	@Override
	public Matrix applyErrorFunction(Matrix input, Matrix target) {
		return input.subtract(target);
	}
}
