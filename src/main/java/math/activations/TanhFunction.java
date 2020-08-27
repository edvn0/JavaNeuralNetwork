package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class TanhFunction implements ActivationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = -1618049951503185298L;
	private static final String NAME = "TANH";

	private double tanh(double a) {
		return Math.tanh(a);
	}

	private double tanhDerivative(double a) {
		return 1 - (a * a);
	}

	@Override
	public Matrix applyFunction(Matrix input) {
		Matrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::tanh);
		return returnDenseMatrix;
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		Matrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::tanhDerivative);
		return returnDenseMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

}
