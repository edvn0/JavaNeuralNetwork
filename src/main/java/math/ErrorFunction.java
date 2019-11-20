package math;

import matrix.Matrix;

public interface ErrorFunction {

	Matrix applyFunction(Matrix input, Matrix target);

}
