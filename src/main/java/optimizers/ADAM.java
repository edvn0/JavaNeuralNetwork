package optimizers;

import math.linearalgebra.Matrix;

import java.util.ArrayList;
import java.util.List;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class ADAM<M> implements Optimizer<M> {

    /**
     *
     */
    private static final long serialVersionUID = -1433313645095435888L;
    private static final String NAME = "Adaptive Moment Estimation";
    private final double lR;
    private final double beta1;
    private final double beta2;
    private List<Matrix<M>> weightM, weightN;
    private List<Matrix<M>> biasM, biasN;

    public ADAM(double alpha, double beta1, double beta2) {
        this.lR = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public List<Matrix<M>> changeWeights(final List<Matrix<M>> weights, final List<Matrix<M>> deltas) {
        log.info("{}",getAdamDeltas(weights, deltas, this.weightM, this.weightN));
        return getAdamDeltas(weights, deltas, this.weightM, this.weightN);
    }

    @Override
    public List<Matrix<M>> changeBiases(final List<Matrix<M>> biases, final List<Matrix<M>> deltas) {
        return getAdamDeltas(biases, deltas, this.biasM, this.biasN);
    }

    private List<Matrix<M>> getAdamDeltas(final List<Matrix<M>> inParams, final List<Matrix<M>> paramDeltas, final List<Matrix<M>> M,
                                          final List<Matrix<M>> N) {
        List<Matrix<M>> newOut = new ArrayList<>(inParams.size());

        for (int i = 0; i < inParams.size(); i++) {
            int exponent = i + 1;
            Matrix<M> mHat;
            Matrix<M> vHat;
            if (M.get(i) != null && N.get(i) != null) {
                // m = beta_1 * m + (1 - beta_1) * g
                // v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
                M.set(i, M.get(i).multiply(beta1).add(paramDeltas.get(i)).multiply((1 - beta1)));
                N.set(i, N.get(i).multiply(beta2).add(paramDeltas.get(i).multiply(paramDeltas.get(i)).multiply((1 - beta2))));
            } else {
                M.set(i, paramDeltas.get(i).multiply((1 - beta1)));
                N.set(i, paramDeltas.get(i).multiply(paramDeltas.get(i).multiply((1 - beta2))));
            }
            mHat = M.get(i).divide((1 - Math.pow(beta1, exponent)));
            vHat = N.get(i).divide((1 - Math.pow(beta2, exponent)));
            Matrix<M> deNom = vHat.mapElements(Math::sqrt).add(10e-8);
            Matrix<M> num = mHat.multiply(this.lR);
            Matrix<M> adam = num.divide(deNom);
            newOut.set(i, inParams.get(i).subtract(adam));
        }
        return newOut;
    }

    @Override
    public void initializeOptimizer(int layers, Matrix<M> weightSeed, Matrix<M> biasSeed) {
        this.weightM = new ArrayList<>(layers);
        this.weightN = new ArrayList<>(layers);
        this.biasM = new ArrayList<>(layers);
        this.biasN = new ArrayList<>(layers);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
