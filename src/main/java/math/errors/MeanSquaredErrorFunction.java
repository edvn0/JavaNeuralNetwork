package math.errors;

import matrix.Matrix;

public class MeanSquaredErrorFunction implements ErrorFunction {

	@Override
	public Matrix applyErrorFunction(Matrix in, Matrix corr) {
		return in.subtract(corr);
	}
}
