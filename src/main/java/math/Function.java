package math;

import matrix.Matrix;

public interface Function {

	Matrix applyFunction(Matrix in, Matrix corr);

	Matrix applyDerivative(Matrix in, Matrix corr);

}
