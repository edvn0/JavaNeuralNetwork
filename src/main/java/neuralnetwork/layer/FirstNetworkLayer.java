package neuralnetwork.layer;

import math.activations.LinearFunction;

public class FirstNetworkLayer<M> extends NetworkLayer<M> {

    public FirstNetworkLayer(LinearFunction<M> activationFunction) {
        super(activationFunction, 0);

    }

}
