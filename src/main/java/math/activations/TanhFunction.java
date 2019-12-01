package math.activations;

import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class TanhFunction implements ActivationFunction {

	private static final String NAME = "TANH";

	private double tanh(double a) {
		return Math.tanh(a);
	}

	private double tanhDerivative(double a) {
		return 1 - (a * a);
	}

	public TanhFunction() {

	}

	@Override
	public DenseMatrix applyFunction(DenseMatrix input) {
		DenseMatrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::tanh);
		return returnDenseMatrix;
	}

	@Override
	public DenseMatrix applyDerivative(DenseMatrix input) {
		DenseMatrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::tanhDerivative);
		return returnDenseMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("TanhFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
