package math.activations;

import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class SigmoidFunction implements ActivationFunction {

	private static final String NAME = "SIGMOID";

	private double sigmoid(double in) {
		return 1 / (1 + Math.exp(-in));
	}

	private double sigmoidDerivative(double a) {
		return a * (1 - a);
	}

	public SigmoidFunction() {

	}

	@Override
	public DenseMatrix applyFunction(DenseMatrix input) {
		DenseMatrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::sigmoid);
		return returnDenseMatrix;
	}

	@Override
	public DenseMatrix applyDerivative(DenseMatrix input) {
		DenseMatrix returnDenseMatrix = input;
		returnDenseMatrix = MatrixUtilities.map(returnDenseMatrix, this::sigmoidDerivative);
		return returnDenseMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("SigmoidFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
