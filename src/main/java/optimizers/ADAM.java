package optimizers;

import org.jetbrains.annotations.NotNull;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class ADAM implements Optimizer {

	private DenseMatrix[] weightM, weightN;
	private DenseMatrix[] biasM, biasN;
	private double lR, beta1, beta2;

	public ADAM(double alpha, double beta1, double beta2) {
		this.lR = alpha;
		this.beta1 = beta1;
		this.beta2 = beta2;
	}

	@Override
	public DenseMatrix[] changeWeights(final DenseMatrix[] weights, final DenseMatrix[] deltas) {
		return getAdamDeltas(weights, deltas, this.weightM, this.weightN);
	}

	@Override
	public DenseMatrix[] changeBiases(final DenseMatrix[] biases, final DenseMatrix[] deltas) {
		return getAdamDeltas(biases, deltas, this.biasM, this.biasN);
	}

	@NotNull
	private DenseMatrix[] getAdamDeltas(final DenseMatrix[] inParams,
		final DenseMatrix[] paramDeltas,
		final DenseMatrix[] M,
		final DenseMatrix[] N) {
		DenseMatrix[] newOut = new DenseMatrix[inParams.length];

		for (int i = 0; i < inParams.length; i++) {
			int index = i + 1;
			DenseMatrix mHat, vHat;
			if (M[i] != null && N[i] != null) {
				// m = beta_1 * m + (1 - beta_1) * g
				// v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
				M[i] = (DenseMatrix)
					M[i]
						.times(beta1)
						.plus(paramDeltas[i]
							.times((1 - beta1)));
				N[i] = (DenseMatrix)
					N[i]
						.times(beta2)
						.plus(paramDeltas[i]
							.times(paramDeltas[i])
							.times((1 - beta2)));
			} else {
				M[i] = (DenseMatrix) paramDeltas[i].times((1 - beta1));
				N[i] = (DenseMatrix) paramDeltas[i].times(paramDeltas[i]).times((1 - beta2));
			}
			mHat = (DenseMatrix) M[i].divide((1 - Math.pow(beta1, index)));
			vHat = (DenseMatrix) N[i].divide((1 - Math.pow(beta2, index)));
			DenseMatrix deNom = (DenseMatrix) vHat.sqrt(Ret.NEW).plus(10e-8);
			DenseMatrix num = (DenseMatrix) mHat.times(this.lR);
			DenseMatrix adam = (DenseMatrix) num.divide(deNom);
			newOut[i] = (DenseMatrix) inParams[i].minus(adam);
		}
		return newOut;
	}

	@Override
	public void initializeOptimizer(int layers) {
		this.weightM = new DenseMatrix[layers];
		this.weightN = new DenseMatrix[layers];
		this.biasM = new DenseMatrix[layers];
		this.biasN = new DenseMatrix[layers];
	}
}
