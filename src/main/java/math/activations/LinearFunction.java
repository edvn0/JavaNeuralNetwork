package math.activations;

import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class LinearFunction implements ActivationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 5919973604842732692L;

	public double linear(double a) {
		return a;
	}

	public double linearDerivative(double a) {
		return 1;
	}

	public LinearFunction() {

	}

	@Override
	public DenseMatrix applyFunction(DenseMatrix input) {
		return MatrixUtilities.map(input, this::linear);// input.map(this::linear);
	}

	@Override
	public DenseMatrix applyDerivative(DenseMatrix input) {
		return MatrixUtilities.map(input, this::linearDerivative);
	}

	@Override
	public String getName() {
		return "LIN";
	}
}
