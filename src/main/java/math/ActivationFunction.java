package math;

import matrix.Matrix;

public interface ActivationFunction {

	String SIGMOID = "SIGMOID";
	String RELU = "RELU";
	String TANH = "TANH";

	Matrix functionToMatrix(Matrix input);

	Matrix derivativeToMatrix(Matrix input);

	String getName();
}
