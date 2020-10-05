package neuralnetwork.layer;

import math.activations.ActivationFunction;
import math.linearalgebra.Matrix;

public class NetworkLayer<M> {

    private final ActivationFunction<M> activationFunction;
    private final int layerIndex;
    private final int neurons;
    // Represents the data after activating this layer
    private final ThreadLocal<Matrix<M>> activation;
    private Matrix<M> weights;
    private Matrix<M> bias;

    public NetworkLayer(ActivationFunction<M> activationFunction, int layerIndex, int neurons) {
        this.activationFunction = activationFunction;
        this.layerIndex = layerIndex;
        this.activation = new ThreadLocal<>();
        this.neurons = neurons;
    }

    public Matrix<M> calculate(Matrix<M> in) {
        if (layerIndex == 0) {
            this.activation.set(in);
            return in;
        } else {
            Matrix<M> out = activationFunction.function(this.weights.multiply(in).add(bias));
            this.activation.set(out);
            return out;
        }
    }
}
