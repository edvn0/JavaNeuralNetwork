package math;

import matrix.Matrix;

public class MeanSquaredErrorFunction implements ErrorFunction {

	@Override
	public Matrix applyFunction(Matrix in, Matrix corr) {
		return in.subtract(corr);
	}
}
