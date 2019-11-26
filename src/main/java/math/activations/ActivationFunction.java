package math.activations;

import java.io.Serializable;
import matrix.Matrix;

public interface ActivationFunction extends Serializable {

	String SIGMOID = "SIGMOID";
	String RELU = "RELU";
	String TANH = "TANH";
	String LIN = "LIN";

	Matrix applyFunction(Matrix input);

	Matrix applyDerivative(Matrix input);

	String getName();

	String toString();
}
