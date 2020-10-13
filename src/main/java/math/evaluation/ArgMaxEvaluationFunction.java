package math.evaluation;

import java.util.LinkedHashMap;
import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

/**
 * Evaluation function which returns a correct value iff the argmax of the predicted data is the
 * label
 */
public class ArgMaxEvaluationFunction<M> implements EvaluationFunction<M> {

	private static final String NAME = "Argmax Evaluation";

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double evaluatePrediction(List<NetworkInput<M>> toEvaluate) {
		int correct = 0;
		for (NetworkInput<M> networkInput : toEvaluate) {
			Matrix<M> data = networkInput.getData();
			int correctLabels = networkInput.getLabel().argMax();
			int val = data.argMax();

			if (correctLabels == val) {
				correct++;
			}

		}
		return (double) correct / toEvaluate.size();
	}

	@Override
	public String name() {
		return NAME;
	}

	@Override
	public LinkedHashMap<String, Double> params() {
		return null;
	}

	@Override
	public void init(double... in) {
	}
}
