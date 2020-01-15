package math.evaluation;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class ArgMaxEvaluationFunction implements EvaluationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 3730260463010183881L;

	public ArgMaxEvaluationFunction() {
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double evaluatePrediction(List<NetworkInput> toEvaluate) {
		int correct = 0;
		for (NetworkInput networkInput : toEvaluate) {
			// data[i] = {data, correctLabels}
			DenseMatrix data = networkInput.getData();
			int correctLabels = MatrixUtilities.argMax(networkInput.getLabel());
			int val = MatrixUtilities.argMax(data);

			if (correctLabels == val) {
				correct++;
			}

		}
		return (double) correct / toEvaluate.size();
	}

}
