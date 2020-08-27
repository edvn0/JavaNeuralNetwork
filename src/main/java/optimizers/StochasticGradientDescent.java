package optimizers;

import org.ujmp.core.Matrix;

public class StochasticGradientDescent implements Optimizer {

	/**
	 *
	 */
	private static final long serialVersionUID = 3872768343431121671L;
	private double learningRate;

	public StochasticGradientDescent(double l) {
		this.learningRate = l;
	}

	@Override
	public Matrix[] changeWeights(final Matrix[] weights, final Matrix[] deltas) {
		Matrix[] out = new Matrix[weights.length];
		for (int i = 0; i < weights.length; i++) {
			deltas[i] = deltas[i].times(this.learningRate);
			out[i] = weights[i].minus(deltas[i]);
		}
		return out;
	}

	@Override
	public Matrix[] changeBiases(final Matrix[] biases, final Matrix[] deltas) {
		Matrix[] out = new Matrix[biases.length];
		for (int i = 0; i < biases.length; i++) {
			deltas[i] = deltas[i].times(this.learningRate);
			out[i] = biases[i].minus(deltas[i]);
		}
		return out;
	}

	@Override
	public void initializeOptimizer(final int layers) {
		// We need no intialisation for this optimiser.
	}
}
