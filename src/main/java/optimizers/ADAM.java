package optimizers;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.util.ArrayList;
import java.util.List;

import lombok.extern.slf4j.Slf4j;

public class ADAM implements Optimizer {

    /**
     *
     */
    private static final long serialVersionUID = -1433313645095435888L;
    private static final String NAME = "Adaptive Moment Estimation";
    private static final double epsilon = 1e-8;
    private final double lR;
    private final double beta1;
    private final double beta2;
    private List<OjAlgoMatrix> weightM, weightN;
    private List<OjAlgoMatrix> biasM, biasN;

    public ADAM(double alpha, double beta1, double beta2) {
        this.lR = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public List<OjAlgoMatrix> changeWeights(final List<OjAlgoMatrix> weights, final List<OjAlgoMatrix> deltas) {
        return getAdamDeltas(weights, deltas, this.weightM, this.weightN);
    }

    @Override
    public List<OjAlgoMatrix> changeBiases(final List<OjAlgoMatrix> biases, final List<OjAlgoMatrix> deltas) {
        return getAdamDeltas(biases, deltas, this.biasM, this.biasN);
    }

    private List<OjAlgoMatrix> getAdamDeltas(final List<OjAlgoMatrix> inParams, final List<OjAlgoMatrix> paramDeltas,
            final List<OjAlgoMatrix> M, final List<OjAlgoMatrix> N) {
        List<OjAlgoMatrix> newOut = new ArrayList<>(inParams.size());

        for (int i = 0; i < inParams.size(); i++) {
            newOut.add(i, null);
        }

        for (int i = 0; i < inParams.size(); i++) {
            int exponent = i + 1;
            OjAlgoMatrix mHat;
            OjAlgoMatrix vHat;
            if (M.get(i) != null && N.get(i) != null) {
                // m = beta_1 * m + (1 - beta_1) * g
                // v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
                OjAlgoMatrix m = M.get(i).multiply(beta1).add(paramDeltas.get(i).multiply((1 - beta1)));
                OjAlgoMatrix v = N.get(i).multiply(beta2)
                        .add(paramDeltas.get(i).hadamard(paramDeltas.get(i)).multiply((1 - beta2)));
                M.set(i, m);
                N.set(i, v);
            } else {
                M.set(i, paramDeltas.get(i).multiply(1 - beta1));
                OjAlgoMatrix fix = paramDeltas.get(i).hadamard(paramDeltas.get(i)).multiply(1 - beta2);
                N.set(i, fix);
            }
            mHat = M.get(i).divide((1 - Math.pow(beta1, exponent)));
            vHat = N.get(i).divide((1 - Math.pow(beta2, exponent)));
            OjAlgoMatrix deNom = vHat.mapElements(Math::sqrt).add(epsilon);
            OjAlgoMatrix num = mHat.multiply(this.lR);
            OjAlgoMatrix adam = num.divide(deNom);
            newOut.set(i, inParams.get(i).subtract(adam));
        }
        return newOut;
    }

    @Override
    public void initializeOptimizer(int layers, OjAlgoMatrix weightSeed, OjAlgoMatrix biasSeed) {
        this.weightM = new ArrayList<>(layers);
        this.weightN = new ArrayList<>(layers);
        this.biasM = new ArrayList<>(layers);
        this.biasN = new ArrayList<>(layers);

        for (int i = 0; i < layers; i++) {
            this.weightM.add(null);
            this.weightN.add(null);
            this.biasM.add(null);
            this.biasN.add(null);
        }
    }

    @Override
    public String toString() {
        return NAME;
    }
}
