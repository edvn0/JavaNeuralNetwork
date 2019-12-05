package math.activations;

import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class ReluFunction implements ActivationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 5116981892314745595L;
	private static final String NAME = "RELU";

	private double relu(double a) {
		return a > 0 ? a : 0;
	}

	private double reluDerivative(double a) {
		return a > 0 ? 1 : 0;
	}

	public ReluFunction() {
	}

	@Override
	public DenseMatrix applyFunction(DenseMatrix input) {
		DenseMatrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::relu);
		return returnMatrix;
	}

	@Override
	public DenseMatrix applyDerivative(DenseMatrix input) {
		DenseMatrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::reluDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("ReluFunction{");
		sb.append("name='").append(getName()).append('\'');
		sb.append('}');
		return sb.toString();
	}
}
