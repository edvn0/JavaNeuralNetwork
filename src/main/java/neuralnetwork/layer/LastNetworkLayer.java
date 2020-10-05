package neuralnetwork.layer;

import math.activations.ActivationFunction;

public class LastNetworkLayer<M> extends NetworkLayer<M> {

    public LastNetworkLayer(ActivationFunction<M> activationFunction, int layerIndex, int neurons) {
        super(activationFunction, layerIndex, neurons);
    }

}
