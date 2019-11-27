package math.activations;

import java.io.Serializable;
import matrix.Matrix;

public interface ActivationFunction extends Serializable {

	String SIGMOID = "SIGMOID";
	String RELU = "RELU";
	String TANH = "TANH";
	String LIN = "LIN";

	Matrix applyFunction(Matrix input, Matrix corr);

	Matrix applyDerivative(Matrix input, Matrix correct);

	String getName();

	String toString();
}
