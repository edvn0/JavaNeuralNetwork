package optimizers;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.util.ArrayList;
import java.util.List;

public class Momentum implements Optimizer {

    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private static final String NAME = "Momentum";
    private final double lR;
    private final double momentumRate;

    private List<OjAlgoMatrix> lastDeltaWeights, lastDeltaBiases;

    public Momentum(double lR, double momentum) {
        this.lR = lR;
        this.momentumRate = momentum;

    }

    @Override
    public List<OjAlgoMatrix> changeWeights(final List<OjAlgoMatrix> weights, final List<OjAlgoMatrix> deltaWeights) {
        return getMomentumDeltas(weights, deltaWeights, lastDeltaWeights);
    }

    @Override
    public List<OjAlgoMatrix> changeBiases(final List<OjAlgoMatrix> biases, final List<OjAlgoMatrix> deltaBiases) {
        return getMomentumDeltas(biases, deltaBiases, lastDeltaBiases);
    }

    private List<OjAlgoMatrix> getMomentumDeltas(final List<OjAlgoMatrix> in, final List<OjAlgoMatrix> deltaIns,
            final List<OjAlgoMatrix> lastDeltas) {
        List<OjAlgoMatrix> newOut = new ArrayList<>();
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
    public void initializeOptimizer(final int layers, OjAlgoMatrix weightSeed, OjAlgoMatrix biasSeed) {
        lastDeltaWeights = new ArrayList<>(layers);
        lastDeltaBiases = new ArrayList<>(layers);
    }

    @Override
    public String toString() {
        return NAME;
    }
}
