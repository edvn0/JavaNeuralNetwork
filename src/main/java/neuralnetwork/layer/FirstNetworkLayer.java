package neuralnetwork.layer;

import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;

public class FirstNetworkLayer extends NetworkLayer {

    public FirstNetworkLayer(ActivationFunction activationFunction, int layerIndex, int neurons) {
        super(new DoNothingFunction(), layerIndex, neurons);
    }

}
