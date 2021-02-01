package math.costfunctions;

import java.util.List;
import math.linearalgebra.Matrix;
import neuralnetwork.inputs.NetworkInput;

/**
 * The cost of the neural network has to be able to encode information about how "incorrect" its
 * predictions have been, and at the same time represent the gradient of the multidimensional
 * function of the network.
 */
public interface CostFunction<M> {

	/**
	 * The cost function which maps the space of validation set to a scalar (the cost)
	 *
	 * @param tData validation data
	 *
	 * @return a scalar representing the cost of the neural network.
	 */
	double calculateCostFunction(List<NetworkInput<M>> tData);

	double calculateSingle(NetworkInput<M> data);

	/**
	 * Calculate the gradient of the cost function with respect to the last layer
	 *
	 * @param predicted the activations of the last layer
	 * @param target    the correct vector
	 *
	 * @return the gradient of the cost function
	 */
	Matrix<M> applyCostFunctionGradient(Matrix<M> predicted, Matrix<M> target);

	String name();

}
