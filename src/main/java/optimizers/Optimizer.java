package optimizers;

import java.io.Serializable;
import org.ujmp.core.Matrix;

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
	 *
	 * @return weights representing the iteration of the strategy
	 */
	Matrix[] changeWeights(Matrix[] weights, Matrix[] deltas);

	/**
	 * Changes the networks biases (immutably, returns a new set of biases) with
	 * respect to the strategy
	 *
	 * @param biases the biases of the network
	 * @param deltas the gradients provided by backpropagation and normalisation
	 *
	 * @return biases representing the iteration of the strategy
	 */
	Matrix[] changeBiases(Matrix[] biases, Matrix[] deltas);

	/**
	 * Some optimizers need to initialise some base case parameters, here you do
	 * that.
	 *
	 * @param layers how many layers are in the network?
	 */
	void initializeOptimizer(int layers);

}
