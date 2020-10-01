package optimizers;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.util.ArrayList;
import java.util.List;


public class StochasticGradientDescent implements Optimizer {

    /**
     *
     */
    private static final long serialVersionUID = 3872768343431121671L;
    private static final String NAME = "Stochastic Gradient Descent";
    private final double learningRate;

    public StochasticGradientDescent(double l) {
        this.learningRate = l;
    }

    @Override
    public List<OjAlgoMatrix> changeWeights(final List<OjAlgoMatrix> weights, final List<OjAlgoMatrix> deltas) {
        List<OjAlgoMatrix> matrixList = new ArrayList<>();

        for (int i = 0; i < weights.size(); i++) {
            OjAlgoMatrix newValue = deltas.get(i).multiply(this.learningRate);
            matrixList.add(i, weights.get(i).subtract(newValue));
        }

        return matrixList;
    }

    @Override
    public List<OjAlgoMatrix> changeBiases(final List<OjAlgoMatrix> biases, final List<OjAlgoMatrix> deltas) {
        List<OjAlgoMatrix> matrixList = new ArrayList<>();
        for (int i = 0; i < biases.size(); i++) {
            OjAlgoMatrix newValue = deltas.get(i).multiply(this.learningRate);
            matrixList.add(i, biases.get(i).subtract(newValue));
        }
        return matrixList;
    }

    @Override
    public void initializeOptimizer(final int layers, OjAlgoMatrix weightSeed, OjAlgoMatrix biasSeed) {
        // We need no intialisation for this optimiser.
    }

    @Override
    public String toString() {
        return NAME;
    }
}
