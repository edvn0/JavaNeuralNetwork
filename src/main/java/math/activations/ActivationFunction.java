package math.activations;

import java.io.Serializable;
import org.ujmp.core.DenseMatrix;

public interface ActivationFunction extends Serializable {

	DenseMatrix applyFunction(DenseMatrix input);

	DenseMatrix applyDerivative(DenseMatrix input);

	default DenseMatrix derivativeOnInput(DenseMatrix input, DenseMatrix out) {
		return (DenseMatrix) out.times(applyDerivative(input));
	}

	String getName();


	String toString();
}
