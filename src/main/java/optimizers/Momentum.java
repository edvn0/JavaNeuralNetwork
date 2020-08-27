package optimizers;

import org.ujmp.core.Matrix;

public class Momentum implements Optimizer {

	private double lR;
	private double momentumRate;

	private Matrix[] lastDeltaWeights, lastDeltaBiases;

	public Momentum(double lR, double momentum) {
		this.lR = lR;
		this.momentumRate = momentum;

	}

	@Override
	public Matrix[] changeWeights(final Matrix[] weights, final Matrix[] deltaWeights) {
		return getMomentumDeltas(weights, deltaWeights, lastDeltaWeights);
	}

	@Override
	public Matrix[] changeBiases(final Matrix[] biases, final Matrix[] deltaBiases) {
		return getMomentumDeltas(biases, deltaBiases, lastDeltaBiases);
	}

	private Matrix[] getMomentumDeltas(final Matrix[] in, final Matrix[] deltaIns, final Matrix[] lastDeltas) {
		Matrix[] newOut = new Matrix[in.length];
		for (int i = 0; i < in.length; i++) {
			if (lastDeltas[i] == null) {
				lastDeltas[i] = deltaIns[i].times(this.lR);
			} else {
				lastDeltas[i] = lastDeltas[i].times(momentumRate).plus(deltaIns[i].times(this.lR));
			}
			newOut[i] = in[i].minus(lastDeltas[i]);
		}
		return newOut;
	}

	@Override
	public void initializeOptimizer(final int layers) {
		lastDeltaWeights = new Matrix[layers];
		lastDeltaBiases = new Matrix[layers];
	}
}
