package math.activations;

import java.io.Serializable;
import math.Function;
import matrix.Matrix;

public interface ActivationFunction extends Function, Serializable {

	String SIGMOID = "SIGMOID";
	String RELU = "RELU";
	String TANH = "TANH";
	String LIN = "LIN";

	Matrix applyFunction(Matrix input, Matrix corr);

	Matrix applyDerivative(Matrix input, Matrix correct);

	String getName();

	String toString();
}
