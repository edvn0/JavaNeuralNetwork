package math.activations;

import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class LeakyReluFunction implements ActivationFunction {

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
	public DenseMatrix applyFunction(final DenseMatrix input) {
		DenseMatrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::leakyRelu);
		return returnMatrix;
	}

	@Override
	public DenseMatrix applyDerivative(final DenseMatrix input) {
		DenseMatrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::leakyReluDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return null;
	}
}
