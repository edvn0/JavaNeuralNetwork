package optimizers;

import org.jetbrains.annotations.NotNull;
import org.ujmp.core.DenseMatrix;

public class Momentum implements Optimizer {

	private double lR, momentum;

	private DenseMatrix[] lastDeltaWeights, lastDeltaBiases;

	public Momentum(double lR, double momentum) {
		this.lR = lR;
		this.momentum = momentum;

	}

	@Override
	public DenseMatrix[] changeWeights(final DenseMatrix[] weights,
		final DenseMatrix[] deltaWeights) {
		return getMomentumDeltas(weights, deltaWeights, lastDeltaWeights);
	}

	@Override
	public DenseMatrix[] changeBiases(final DenseMatrix[] biases, final DenseMatrix[] deltaBiases) {
		return getMomentumDeltas(biases, deltaBiases, lastDeltaBiases);
	}

	@NotNull
	private DenseMatrix[] getMomentumDeltas(final DenseMatrix[] in, final DenseMatrix[] deltaIns,
		final DenseMatrix[] lastDeltas) {
		DenseMatrix[] newOut = new DenseMatrix[in.length];
		for (int i = 0; i < in.length; i++) {
			if (lastDeltas[i] == null) {
				lastDeltas[i] = (DenseMatrix) deltaIns[i].times(this.lR);
			} else {
				lastDeltas[i] = (DenseMatrix) lastDeltas[i].times(momentum)
					.plus(deltaIns[i].times(this.lR));
			}
			newOut[i] = (DenseMatrix) in[i].minus(lastDeltas[i]);
		}
		return newOut;
	}

	@Override
	public void initializeOptimizer(final int layers) {
		lastDeltaWeights = new DenseMatrix[layers];
		lastDeltaBiases = new DenseMatrix[layers];
	}
}
