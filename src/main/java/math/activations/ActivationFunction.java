package math.activations;

import java.io.Serializable;
import org.ujmp.core.Matrix;

public interface ActivationFunction extends Serializable {

	Matrix applyFunction(Matrix input);

	Matrix applyDerivative(Matrix input);

	default Matrix derivativeOnInput(Matrix input, Matrix out) {
		return out.times(applyDerivative(input));
	}

	String getName();

	String toString();
}
