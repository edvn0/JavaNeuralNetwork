package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class SigmoidFunction implements ActivationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 8268667665334669223L;
	private static final String NAME = "SIGMOID";

	private double sigmoid(double in) {
		return 1 / (1 + Math.exp(-in));
	}

	private double sigmoidDerivative(double a) {
		return a * (1 - a);
	}

	@Override
	public Matrix applyFunction(Matrix input) {
		Matrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::sigmoid);
		return returnDenseMatrix;
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		Matrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::sigmoidDerivative);
		return returnDenseMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}
}
