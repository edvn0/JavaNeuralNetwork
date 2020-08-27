package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class LinearFunction implements ActivationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 5919973604842732692L;

	private static final int LIN_DER = 1;

	public double linear(double a) {
		return a;
	}

	public double linearDerivative(double a) {
		return LIN_DER;
	}

	@Override
	public Matrix applyFunction(Matrix input) {
		return MatrixUtilities.map(input, this::linear);
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		return MatrixUtilities.map(input, this::linearDerivative);
	}

	@Override
	public String getName() {
		return "LIN";
	}
}
