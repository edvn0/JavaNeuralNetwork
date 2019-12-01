package math.activations;

import java.io.Serializable;
import org.ujmp.core.DenseMatrix;

public interface ActivationFunction extends Serializable {

	String SIGMOID = "SIGMOID";
	String RELU = "RELU";
	String TANH = "TANH";
	String LIN = "LIN";

	DenseMatrix applyFunction(DenseMatrix input);

	DenseMatrix applyDerivative(DenseMatrix input);

	String getName();

	String toString();
}
