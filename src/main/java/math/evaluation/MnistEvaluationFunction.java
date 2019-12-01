package math.evaluation;

import java.util.List;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

public class MnistEvaluationFunction implements EvaluationFunction {


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
		return Matrix.Factory.importFromArray(new int[][]{{correct}});
	}

	private int getLabel(DenseMatrix matrix) {
		int i = 0;
		for (double[] d : matrix.toDoubleArray()) {
			int val = (int) d[0];
			if (val == 1) {
				return i;
			}
			i++;
		}
		return -1;
	}

	private int maxLabel(DenseMatrix fedForward) {
		double[] data = fedForward.toDoubleArray()[0];
		int index = 0;
		double max = data[0];
		for (int i = 1; i < data.length; i++) {
			if (data[i] > max) {
				max = data[i];
				index = i;
			}
		}
		return index;
	}
}
