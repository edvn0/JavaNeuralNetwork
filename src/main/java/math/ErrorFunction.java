package math;

import matrix.Matrix;

public interface ErrorFunction {

	Matrix applyErrorFunction(Matrix input, Matrix target);

}
