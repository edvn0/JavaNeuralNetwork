package neuralnetwork.layer;

import math.activations.ActivationFunction;
import math.linearalgebra.Matrix;
import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class NetworkLayer {

    private final ActivationFunction activationFunction;
    private final int layerIndex;
    private final int neurons;
    // Represents the data after activating this layer
    private final ThreadLocal<OjAlgoMatrix> activation;
    private OjAlgoMatrix weights;
    private OjAlgoMatrix bias;

    public NetworkLayer(ActivationFunction activationFunction, int layerIndex, int neurons) {
        this.activationFunction = activationFunction;
        this.layerIndex = layerIndex;
        this.activation = new ThreadLocal<>();
        this.neurons = neurons;
    }

    public OjAlgoMatrix calculate(OjAlgoMatrix in) {
        if (layerIndex == 0) {
            this.activation.set(in);
            return in;
        } else {
            OjAlgoMatrix out = activationFunction.function(this.weights.multiply(in).add(bias));
            this.activation.set(out);
            return out;
        }
    }
}
