package math.error_functions;

import neuralnetwork.inputs.NetworkInput;
import org.ujmp.core.Matrix;

import java.io.Serializable;
import java.util.List;

/**
 * The cost of the neural network has to be able to encode information about how
 * "incorrect" its predictions have been, and at the same time represent the
 * gradient of the multidimensional function of the network.
 */
public interface CostFunction extends Serializable {

    /**
     * The cost function which maps the space of validation set to a scalar (the
     * cost)
     *
     * @param tData validation data
     * @return a scalar representing the cost of the neural network.
     */
    double calculateCostFunction(List<NetworkInput> tData);

    /**
     * Calculate the gradient of the cost function with respect to the last layer
     *
     * @param in      the activations of the last layer
     * @param correct the correct vector
     * @return the gradient of the cost function
     */
    Matrix applyCostFunctionGradient(Matrix in, Matrix correct);

}
