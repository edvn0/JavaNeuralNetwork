package math.evaluation;

import java.util.List;
import neuralnetwork.inputs.NetworkInput;
import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

/**
 * Evaluation function which returns a correct value iff the argmax of the
 * predicted data is the label
 */
public class ArgMaxEvaluationFunction implements EvaluationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 3730260463010183881L;

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double evaluatePrediction(List<NetworkInput> toEvaluate) {
		int correct = 0;
		for (NetworkInput networkInput : toEvaluate) {
			Matrix data = networkInput.getData();
			int correctLabels = MatrixUtilities.argMax(networkInput.getLabel());
			int val = MatrixUtilities.argMax(data);

			if (correctLabels == val) {
				correct++;
			}

		}
		return (double) correct / toEvaluate.size();
	}

}
