package math;

import matrix.Matrix;

public interface ActivationFunction {

	String SIGMOID = "SIGMOID";
	String RELU = "RELU";
	String TANH = "TANH";
	String LIN = "LIN";

	Matrix applyFunction(Matrix input);

	Matrix applyDerivative(Matrix input);

	String getName();

	String toString();
}
