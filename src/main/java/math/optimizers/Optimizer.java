package math.optimizers;

import java.util.List;
import math.linearalgebra.Matrix;
import utilities.serialise.NetworkSerializable;

/**
 * The optimizer for the gradient descent, represents some strategy for the neural network.
 */
public interface Optimizer<M> extends NetworkSerializable<String, Double> {

	void init(double... in);

	/**
	 * Changes the networks weights (immutably, returns a new set of weights) with respect to the
	 * strategy
	 *
	 * @param weights the weights of the network
	 * @param deltas  the gradients provided by backpropagation and normalisation
	 *
	 * @return weights representing the iteration of the strategy
	 */
	List<Matrix<M>> changeWeights(List<Matrix<M>> weights, List<Matrix<M>> deltas);

	/**
	 * Changes the networks biases (immutably, returns a new set of biases) with respect to the
	 * strategy
	 *
	 * @param biases the biases of the network
	 * @param deltas the gradients provided by backpropagation and normalisation
	 *
	 * @return biases representing the iteration of the strategy
	 */
	List<Matrix<M>> changeBiases(List<Matrix<M>> biases, List<Matrix<M>> deltas);

	Matrix<M> changeBias(int layerIndex, Matrix<M> bias, Matrix<M> deltaBias);

	Matrix<M> changeWeight(int layerIndex, Matrix<M> weight, Matrix<M> deltaWeight);

	/**
	 * Some optimizers need to initialise some base case parameters, here you do that.
	 *
	 * @param layers how many layers are in the network?
	 */
	void initializeOptimizer(int layers, Matrix<M> weightSeed, Matrix<M> biasSeed);

	String name();

}
