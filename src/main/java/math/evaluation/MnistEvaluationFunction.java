package math.evaluation;

import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;

public class MnistEvaluationFunction implements EvaluationFunction {


	public MnistEvaluationFunction() {
	}

	@Override
	public Matrix evaluatePrediction(List<NetworkInput> toEvaluate) {
		int correct = 0;
		for (NetworkInput networkInput : toEvaluate) {
			// data[i] = {data, correctLabels}
			Matrix data = networkInput.getData();
			int correctLabels = this.getLabel(networkInput.getLabel());

			int val = this.maxLabel(data);
			if (correctLabels == val) {
				correct++;
			}

		}
		return Matrix.fromArray(new double[]{correct});
	}

	private int getLabel(Matrix matrix) {
		int i = 0;
		for (double[] d : matrix.getData()) {
			int val = (int) d[0];
			if (val == 1) {
				return i;
			}
			i++;
		}
		return -1;
	}

	private int maxLabel(Matrix fedForward) {
		double[] data = fedForward.toArray();
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
