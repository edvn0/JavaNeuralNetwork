package math.optimizers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

import math.linearalgebra.Matrix;
import neuralnetwork.layer.NetworkLayer;

public class StochasticGradientDescent<M> implements Optimizer<M> {

    private static final String NAME = "Stochastic Gradient Descent";
    private double learningRate;

    public StochasticGradientDescent(double l) {
        this.learningRate = l;
    }

    public StochasticGradientDescent() {
    }

    @Override
    public LinkedHashMap<String, Double> params() {
        LinkedHashMap<String, Double> oMap = new LinkedHashMap<>();
        oMap.put("v1", learningRate);
        return oMap;
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
    public String name() {
        return NAME;
    }

    @Override
    public void init(double... in) {
        this.learningRate = in[0];
    }

    @Override
    public void changeBias(int index, NetworkLayer<M> layer, Matrix<M> deltaBias) {
        Matrix<M> newBias = layer.getBias().subtract(deltaBias.multiply(this.learningRate));
        layer.setBias(newBias);
    }

    @Override
    public void changeWeight(int index, NetworkLayer<M> layer, Matrix<M> deltaWeight) {
        Matrix<M> newWeight = layer.getWeight().subtract(deltaWeight.multiply(this.learningRate));
        layer.setBias(newWeight);
    }
}
