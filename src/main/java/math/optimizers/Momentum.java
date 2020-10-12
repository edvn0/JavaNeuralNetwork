package math.optimizers;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

import math.linearalgebra.Matrix;
import neuralnetwork.layer.NetworkLayer;

public class Momentum<M> implements Optimizer<M> {

    private static final String NAME = "Momentum";
    private double lR;
    private double momentumRate;

    private List<Matrix<M>> lastDeltaWeights, lastDeltaBiases;

    public Momentum(double lR, double momentum) {
        this.lR = lR;
        this.momentumRate = momentum;

    }

    public Momentum() {
    }

    @Override
    public LinkedHashMap<String, Double> params() {
        LinkedHashMap<String, Double> oMap = new LinkedHashMap<>();
        oMap.put("v1", lR);
        oMap.put("v2", momentumRate);
        return oMap;
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
    public String name() {
        return NAME;
    }

    @Override
    public void init(double... in) {
        this.lR = in[0];
        this.momentumRate = in[1];

    }

    @Override
    public void changeBias(int layerIndex, NetworkLayer<M> bias, Matrix<M> deltaBias) {
        // TODO Auto-generated method stub

    }

    @Override
    public void changeWeight(int layerIndex, NetworkLayer<M> weight, Matrix<M> deltaWeight) {
        // TODO Auto-generated method stub

    }

}
