package optimizers;

import org.ujmp.core.DenseMatrix;

public class RMSProp implements Optimizer {


	@Override
	public DenseMatrix[] changeWeights(final DenseMatrix[] weights, final DenseMatrix[] deltas) {
		return new DenseMatrix[0];
	}

	@Override
	public DenseMatrix[] changeBiases(final DenseMatrix[] biases, final DenseMatrix[] deltas) {
		return new DenseMatrix[0];
	}

	@Override
	public void initializeOptimizer(final int layers) {

	}
}
