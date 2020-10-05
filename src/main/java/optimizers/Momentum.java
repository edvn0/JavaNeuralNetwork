package optimizers;

import java.util.ArrayList;
import java.util.List;

import math.linearalgebra.Matrix;

public class Momentum<M> implements Optimizer<M> {

    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private static final String NAME = "Momentum";
    private final double lR;
    private final double momentumRate;

    private List<Matrix<M>> lastDeltaWeights, lastDeltaBiases;

    public Momentum(double lR, double momentum) {
        this.lR = lR;
        this.momentumRate = momentum;

    }

    @Override
    public List<Matrix<M>> changeWeights(final List<Matrix<M>> weights, final List<Matrix<M>> deltaWeights) {
        return getMomentumDeltas(weights, deltaWeights, lastDeltaWeights);
    }

    @Override
    public List<Matrix<M>> changeBiases(final List<Matrix<M>> biases, final List<Matrix<M>> deltaBiases) {
        return getMomentumDeltas(biases, deltaBiases, lastDeltaBiases);
    }

    private List<Matrix<M>> getMomentumDeltas(final List<Matrix<M>> in, final List<Matrix<M>> deltaIns,
            final List<Matrix<M>> lastDeltas) {
        List<Matrix<M>> newOut = new ArrayList<>();
        for (int i = 0; i < in.size(); i++) {
            if (lastDeltas.get(i) == null) {
                lastDeltas.set(i, deltaIns.get(i).multiply(this.lR));
            } else {
                lastDeltas.set(i, lastDeltas.get(i)).multiply(momentumRate).add(deltaIns.get(i).multiply(this.lR));
            }
            newOut.add(i, in.get(i).subtract(lastDeltas.get(i)));
        }
        return newOut;
    }

    @Override
    public void initializeOptimizer(final int layers, Matrix<M> weightSeed, Matrix<M> biasSeed) {
        lastDeltaWeights = new ArrayList<>(layers);
        lastDeltaBiases = new ArrayList<>(layers);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
