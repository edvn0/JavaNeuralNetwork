package math.evaluation;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class MnistEvaluationFunction implements EvaluationFunction {

	/**
	 *
	 */
	private static final long serialVersionUID = 3730260463010183881L;

	public MnistEvaluationFunction() {
	}

	@Override
	public DenseMatrix evaluatePrediction(List<NetworkInput> toEvaluate) {
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
		return Matrix.Factory.importFromArray(new int[][] { { correct } });
	}

}
