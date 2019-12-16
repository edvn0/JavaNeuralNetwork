package optimizers;

import org.ujmp.core.DenseMatrix;

public class StochasticGradientDescent implements Optimizer {

	private double learningRate;

	public StochasticGradientDescent(double l) {
		this.learningRate = l;
	}

	@Override
	public DenseMatrix[] changeWeights(final DenseMatrix[] weights, final DenseMatrix[] deltas) {
		DenseMatrix[] out = new DenseMatrix[weights.length];
		for (int i = 0; i < weights.length; i++) {
			deltas[i] = (DenseMatrix) deltas[i].times(this.learningRate);
			out[i] = (DenseMatrix) weights[i].minus(deltas[i]);
		}
		return out;
	}

	@Override
	public DenseMatrix[] changeBiases(final DenseMatrix[] biases, final DenseMatrix[] deltas) {
		DenseMatrix[] out = new DenseMatrix[biases.length];
		for (int i = 0; i < biases.length; i++) {
			deltas[i] = (DenseMatrix) deltas[i].times(this.learningRate);
			out[i] = (DenseMatrix) biases[i].minus(deltas[i]);
		}
		return out;
	}


	@Override
	public void initializeOptimizer(final int layers) {
	}
}
