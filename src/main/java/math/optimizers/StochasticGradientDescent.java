package math.optimizers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import math.linearalgebra.Matrix;
import org.jetbrains.annotations.NotNull;

public class StochasticGradientDescent<M> implements Optimizer<M> {

	private static final String NAME = "Stochastic Gradient Descent";
	private double learningRate;

	public StochasticGradientDescent(double l) {
		this.learningRate = l;
	}

	public StochasticGradientDescent() {
	}

	@Override
	public LinkedHashMap<String, Double> params() {
		LinkedHashMap<String, Double> oMap = new LinkedHashMap<>();
		oMap.put("v1", learningRate);
		return oMap;
	}

	@Override
	public void init(double... in) {
		this.learningRate = in[0];
	}

	@Override
	public List<Matrix<M>> changeWeights(final List<Matrix<M>> weights,
		final List<Matrix<M>> deltas) {
		return sgdDelta(weights, deltas);
	}

	@Override
	public List<Matrix<M>> changeBiases(final List<Matrix<M>> biases,
		final List<Matrix<M>> deltas) {
		return sgdDelta(biases, deltas);
	}

	@Override
	public Matrix<M> changeBias(int index, Matrix<M> bias, Matrix<M> deltaBias) {
		return bias.subtract(deltaBias.multiply(this.learningRate));

	}

	@Override
	public Matrix<M> changeWeight(int index, Matrix<M> weight, Matrix<M> deltaWeight) {
		return weight.subtract(deltaWeight.multiply(this.learningRate));
	}

	@Override
	public void initializeOptimizer(final int layers, Matrix<M> weightSeed, Matrix<M> biasSeed) {
		// We need no intialisation for this optimiser.
	}

	@Override
	public String name() {
		return NAME;
	}

	@NotNull
	private List<Matrix<M>> sgdDelta(final List<Matrix<M>> weights, final List<Matrix<M>> deltas) {
		List<Matrix<M>> matrixList = new ArrayList<>();
		for (int i = 0; i < weights.size(); i++) {
			Matrix<M> newValue = deltas.get(i).multiply(this.learningRate);
			matrixList.add(i, weights.get(i).subtract(newValue));
		}
		return matrixList;
	}
}
