package math;

import matrix.Matrix;

public interface ActivationFunction {

	String SIGMOID = "SIGMOID";
	String RELU = "RELU";
	String TANH = "TANH";

	Matrix applyFunctionToMatrix(Matrix input);

	Matrix applyDerivativeFunctionToMatrix(Matrix input);

	String getName();
}
