package math.evaluation;

import java.util.Arrays;
import java.util.List;
import matrix.Matrix;

public class MnistEvaluationFunction implements EvaluationFunction {

	@Override
	public Matrix evaluatePrediction(List<Matrix[]> toEvaluate) {
		int correct = 0;
		for (int i = 0; i < toEvaluate.size(); i++) {
			// data[i] = {data, correctLabels}
			Matrix data = toEvaluate.get(i)[0];
			Matrix correctLabels = toEvaluate.get(i)[1];

			int val = this.maxLabel(data);
			if (correctLabels.getElement(0, 0) == val) {
				correct++;
			}

		}
		return Matrix.fromArray(new double[]{correct});
	}

	private int maxLabel(Matrix fedForward) {
		double[] data = fedForward.toArray();
		System.out.println(Arrays.toString(data));
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
