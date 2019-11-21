package math.errors;

import matrix.Matrix;

public interface ErrorFunction {

	Matrix applyErrorFunction(Matrix input, Matrix target);

}
