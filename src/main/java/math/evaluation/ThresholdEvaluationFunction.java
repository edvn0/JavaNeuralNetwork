package math.evaluation;

import java.util.LinkedHashMap;
import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

public class ThresholdEvaluationFunction<M> implements EvaluationFunction<M> {

	private static final String NAME = "Threshold Evaluation";
	private double threshold;

	public ThresholdEvaluationFunction(double a) {
		this.threshold = a;
	}

	public ThresholdEvaluationFunction() {
	}

	@Override
	public LinkedHashMap<String, Double> params() {
		LinkedHashMap<String, Double> oMap = new LinkedHashMap<>();
		oMap.put("v1", threshold);
		return oMap;
	}

	@Override
	public void init(double... in) {
		this.threshold = in[0];

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double evaluatePrediction(List<NetworkInput<M>> toEvaluate) {
		int correct = 0;
		for (NetworkInput<M> matrices : toEvaluate) {
			Matrix<M> fed = matrices.getData();
			Matrix<M> corr = matrices.getLabel();

			double fedEl = fed.sum();
			double corrEl = corr.sum();

			if (Math.abs(fedEl - corrEl) < this.threshold) {
				correct++;
			}
		}
		return (double) correct / toEvaluate.size();
	}

	@Override
	public String name() {
		return NAME;
	}
}
