package optimizers;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class ADAM implements Optimizer {

    private final double lR;
    private final double beta1;
    private final double beta2;
    private Matrix[] weightM, weightN;
    private Matrix[] biasM, biasN;

    public ADAM(double alpha, double beta1, double beta2) {
        this.lR = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public Matrix[] changeWeights(final Matrix[] weights, final Matrix[] deltas) {
        return getAdamDeltas(weights, deltas, this.weightM, this.weightN);
    }

    @Override
    public Matrix[] changeBiases(final Matrix[] biases, final Matrix[] deltas) {
        return getAdamDeltas(biases, deltas, this.biasM, this.biasN);
    }

    private Matrix[] getAdamDeltas(final Matrix[] inParams, final Matrix[] paramDeltas, final Matrix[] M,
                                   final Matrix[] N) {
        Matrix[] newOut = new Matrix[inParams.length];

        for (int i = 0; i < inParams.length; i++) {
            int exponent = i + 1;
            Matrix mHat;
            Matrix vHat;
            if (M[i] != null && N[i] != null) {
                // m = beta_1 * m + (1 - beta_1) * g
                // v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
                M[i] = M[i].times(beta1).plus(paramDeltas[i].times((1 - beta1)));
                N[i] = N[i].times(beta2).plus(paramDeltas[i].times(paramDeltas[i]).times((1 - beta2)));
            } else {
                M[i] = paramDeltas[i].times((1 - beta1));
                N[i] = paramDeltas[i].times(paramDeltas[i]).times((1 - beta2));
            }
            mHat = M[i].divide((1 - Math.pow(beta1, exponent)));
            vHat = N[i].divide((1 - Math.pow(beta2, exponent)));
            Matrix deNom = vHat.sqrt(Ret.NEW).plus(10e-8);
            Matrix num = mHat.times(this.lR);
            Matrix adam = num.divide(deNom);
            newOut[i] = inParams[i].minus(adam);
        }
        return newOut;
    }

    @Override
    public void initializeOptimizer(int layers) {
        this.weightM = new Matrix[layers];
        this.weightN = new Matrix[layers];
        this.biasM = new Matrix[layers];
        this.biasN = new Matrix[layers];
    }
}
