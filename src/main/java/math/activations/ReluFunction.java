package math.activations;

import org.ujmp.core.Matrix;
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

	@Override
	public Matrix applyFunction(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::relu);
		return returnMatrix;
	}

	@Override
	public Matrix applyDerivative(Matrix input) {
		Matrix returnMatrix = input;
		returnMatrix = MatrixUtilities.map(returnMatrix, this::reluDerivative);
		return returnMatrix;
	}

	@Override
	public String getName() {
		return NAME;
	}

}
