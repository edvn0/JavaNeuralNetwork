package neuralnetwork.layer;

import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;

public class FirstNetworkLayer<M> extends NetworkLayer<M> {

    public FirstNetworkLayer(ActivationFunction<M> activationFunction, int layerIndex, int neurons) {
        super(new DoNothingFunction<>(), layerIndex, neurons);
    }

}
