package optimizers;

import java.util.ArrayList;
import java.util.List;

import math.linearalgebra.Matrix;

public class StochasticGradientDescent<M> implements Optimizer<M> {

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
    public List<Matrix<M>> changeWeights(final List<Matrix<M>> weights, final List<Matrix<M>> deltas) {
        List<Matrix<M>> matrixList = new ArrayList<>();

        for (int i = 0; i < weights.size(); i++) {
            Matrix<M> newValue = deltas.get(i).multiply(this.learningRate);
            matrixList.add(i, weights.get(i).subtract(newValue));
        }

        return matrixList;
    }

    @Override
    public List<Matrix<M>> changeBiases(final List<Matrix<M>> biases, final List<Matrix<M>> deltas) {
        List<Matrix<M>> matrixList = new ArrayList<>();
        for (int i = 0; i < biases.size(); i++) {
            Matrix<M> newValue = deltas.get(i).multiply(this.learningRate);
            matrixList.add(i, biases.get(i).subtract(newValue));
        }
        return matrixList;
    }

    @Override
    public void initializeOptimizer(final int layers, Matrix<M> weightSeed, Matrix<M> biasSeed) {
        // We need no intialisation for this optimiser.
    }

    @Override
    public String toString() {
        return NAME;
    }
}
