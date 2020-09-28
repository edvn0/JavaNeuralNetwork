package optimizers;

import math.linearalgebra.Matrix;
import org.ujmp.core.calculation.Calculation;

public class ADAM implements Optimizer {

    private final double lR;
    private final double beta1;
    private final double beta2;
    private Matrix<Matrix<Matrix>>[] weightM, weightN;
    private Matrix<Matrix<Matrix>>[] biasM, biasN;

    public ADAM(double alpha, double beta1, double beta2) {
        this.lR = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public Matrix<Matrix<Matrix>>[] changeWeights(final Matrix<Matrix<Matrix>>[] weights, final Matrix<Matrix<Matrix<?>>>[] deltas) {
        return getAdamDeltas(weights, deltas, this.weightM, this.weightN);
    }

    @Override
    public Matrix<Matrix<Matrix>>[] changeBiases(final Matrix<Matrix<Matrix>>[] biases, final Matrix<Matrix<Matrix<?>>>[] deltas) {
        return getAdamDeltas(biases, deltas, this.biasM, this.biasN);
    }

    private Matrix<Matrix<Matrix>>[] getAdamDeltas(final Matrix<Matrix<Matrix>>[] inParams, final Matrix<Matrix<Matrix<?>>>[] paramDeltas, final Matrix<Matrix<Matrix>>[] M,
                                                   final Matrix<Matrix<Matrix>>[] N) {
        Matrix<Matrix<Matrix>>[] newOut = new Matrix<Matrix<Matrix>>[inParams.length];

        for (int i = 0; i < inParams.length; i++) {
            int exponent = i + 1;
            Matrix<Matrix> mHat;
            Matrix<Matrix> vHat;
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
            Matrix<Matrix> deNom = vHat.sqrt(Calculation.Ret.NEW).plus(10e-8);
            Matrix<Matrix> num = mHat.times(this.lR);
            Matrix<Matrix> adam = num.divide(deNom);
            newOut[i] = inParams[i].minus(adam);
        }
        return newOut;
    }

    @Override
    public void initializeOptimizer(int layers) {
        this.weightM = new Matrix<Matrix<Matrix>>[layers];
        this.weightN = new Matrix<Matrix<Matrix>>[layers];
        this.biasM = new Matrix<Matrix<Matrix>>[layers];
        this.biasN = new Matrix<Matrix<Matrix>>[layers];
    }
}
