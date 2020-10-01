package neuralnetwork.layer;

import math.activations.ActivationFunction;

public class LastNetworkLayer extends NetworkLayer {

    public LastNetworkLayer(ActivationFunction activationFunction, int layerIndex, int neurons) {
        super(activationFunction, layerIndex, neurons);
    }

}
