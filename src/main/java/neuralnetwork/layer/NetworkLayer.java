package neuralnetwork.layer;

import math.activations.ActivationFunction;
import org.ujmp.core.Matrix;

public class NetworkLayer {

    private final ActivationFunction activationFunction;
    private final int layerIndex;
    // Represents the data after activating this layer
    private final ThreadLocal<Matrix> activation;
    private Matrix weights;
    private Matrix bias;

    public NetworkLayer(ActivationFunction activationFunction,
                        int layerIndex) {
        this.activationFunction = activationFunction;
        this.layerIndex = layerIndex;
        this.activation = new ThreadLocal<>();
    }

    public Matrix calculate(Matrix in) {
        if (layerIndex == 0) {
            this.activation.set(in);
            return in;
        } else {
            Matrix out = activationFunction.function(this.weights.mtimes(in).plus(bias));
            this.activation.set(out);
            return out;
        }
    }
}
