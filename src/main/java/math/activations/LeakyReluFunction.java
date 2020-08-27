package math.activations;

import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class LeakyReluFunction implements ActivationFunction {

	private static final long serialVersionUID = 1L;
	private double alpha;

	public LeakyReluFunction(double alpha) {
		this.alpha = alpha;
	}

	private double leakyRelu(double v) {
		return v > 0 ? v : alpha * v;
	}

	private double leakyReluDerivative(double v) {
		return v > 0 ? 1 : alpha;
	}

	@Override
	public Matrix applyFunction(final Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::leakyRelu);
		return returnMatrix;
	}

	@Override
	public Matrix applyDerivative(final Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::leakyReluDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return null;
	}
}
