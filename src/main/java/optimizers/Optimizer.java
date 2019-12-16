package optimizers;

import java.io.Serializable;
import org.ujmp.core.DenseMatrix;

/**
 * The optimizer for the gradient descent, represents some strategy for the neural network.
 */
public interface Optimizer extends Serializable {

	/**
	 * Changes the networks weights (immutably, returns a new set of weights) with respect to the
	 * strategy
	 *
	 * @param weights the weights of the network
	 * @param deltas  the gradients provided by backpropagation and normalisation
	 *
	 * @return weights representing the iteration of the strategy
	 */
	DenseMatrix[] changeWeights(DenseMatrix[] weights, DenseMatrix[] deltas);

	/**
	 * Changes the networks biases (immutably, returns a new set of biases) with respect to the
	 * strategy
	 *
	 * @param biases the biases of the network
	 * @param deltas the gradients provided by backpropagation and normalisation
	 *
	 * @return biases representing the iteration of the strategy
	 */
	DenseMatrix[] changeBiases(DenseMatrix[] biases, DenseMatrix[] deltas);

	/**
	 * Some optimizers need to initialise some base case fields, here you do that.
	 *
	 * @param layers how many layers are in the network?
	 */
	void initializeOptimizer(int layers);

}
