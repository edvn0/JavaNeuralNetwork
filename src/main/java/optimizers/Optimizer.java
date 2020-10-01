package optimizers;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

import java.io.Serializable;
import java.util.List;

/**
 * The optimizer for the gradient descent, represents some strategy for the
 * neural network.
 */
public interface Optimizer extends Serializable {

    /**
     * Changes the networks weights (immutably, returns a new set of weights) with
     * respect to the strategy
     *
     * @param weights the weights of the network
     * @param deltas  the gradients provided by backpropagation and normalisation
     * @return weights representing the iteration of the strategy
     */
    List<OjAlgoMatrix> changeWeights(List<OjAlgoMatrix> weights, List<OjAlgoMatrix> deltas);

    /**
     * Changes the networks biases (immutably, returns a new set of biases) with
     * respect to the strategy
     *
     * @param biases the biases of the network
     * @param deltas the gradients provided by backpropagation and normalisation
     * @return biases representing the iteration of the strategy
     */
    List<OjAlgoMatrix> changeBiases(List<OjAlgoMatrix> biases, List<OjAlgoMatrix> deltas);

    /**
     * Some optimizers need to initialise some base case parameters, here you do
     * that.
     *
     * @param layers how many layers are in the network?
     */
    void initializeOptimizer(int layers, OjAlgoMatrix weightSeed, OjAlgoMatrix biasSeed);

    String toString();

}
