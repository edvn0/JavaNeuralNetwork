package neuralnetwork;

import java.io.Serializable;
import math.ActivationFunction;
import math.SigmoidFunction;
import matrix.Matrix;
import neuralnetwork.layer.LayerConnectionList;

public class NeuralNetwork implements Serializable, Trainable {

	private static final long serialVersionUID = 0L;

	private double learningRate;

	private ActivationFunction function;

	private LayerConnectionList layerConnections; // Maps layer 0 to 1, 1 to 2, etc. Will "store" the

	public NeuralNetwork(int inputNodes, int hiddenLayers, int nodesInHiddenLayers,
		int outputNodes, double learningRate) {
		this.layerConnections = new LayerConnectionList(inputNodes, hiddenLayers,
			nodesInHiddenLayers, outputNodes);
		this.learningRate = learningRate;

		this.function = new SigmoidFunction();
	}

	public NeuralNetwork(int inNodes, int hLayers, int nodesInHidden, int oNodes, double learning,
		ActivationFunction f) {
		this(inNodes, hLayers, nodesInHidden, oNodes, learning);
		this.function = f;
	}

	public NeuralNetwork(int inputNodes, int outputNodes) {
		this(inputNodes, 3, 30, outputNodes, 10e-2, new SigmoidFunction());
	}

	@Override
	public void train(double[] in, double[] correct) {
		Matrix inputMatrix = predict(in);
		Matrix correctMatrix = Matrix.fromArray(correct);

		backPropagate(inputMatrix, correctMatrix);
	}

	@Override
	public Matrix predict(double[] in) {
		return feedForward(in);
	}

	/**
	 * Backpropagation, alters weights and biases by gradient descent.
	 *
	 * @param fedForward the predicted value, given by {@link #predict(double[])}
	 * @param correct The correct classification
	 */
	private void backPropagate(Matrix fedForward, Matrix correct) {
		// Calculate errors. Will feed this backwards...
		Matrix errors = correct.subtract(fedForward); // Error vector

		// Last weight, we iterate backwards
		int lastWeight = this.layerConnections.amountOfWeights() - 1;

		for (int i = lastWeight; i >= 0; i--) {
			int layerIndex =
				i + 1; // As opposed to the weight, what layer are we using? - the i+1:th.
			Matrix currentWeight = this.layerConnections.getWeight(i);
			Matrix currentLayer = this.layerConnections.getActiveLayer(layerIndex);
			Matrix currentLayerBias = this.layerConnections.getLayer(layerIndex)
				.getBias(); // Bias of current layer.
			Matrix previousErrors = errors; // To use in this loop, and to not mutate to next iteration

			Matrix currentLayerBiased = currentLayer.add(currentLayerBias); // Add bias to the layer

			// Apply the derivative to each element of the layer,
			// hadamard-product with the errors, multiply with learning rate
			Matrix gradients = this.function.derivativeToMatrix(currentLayerBiased)
				.hadamard(previousErrors)
				.map((e) -> e * this.learningRate);

			// The bias deltas, add the gradients to the deltas.
			Matrix deltaBias = currentLayerBias.add(gradients);

			// Set the bias of this layer to deltaBias
			this.layerConnections.setActiveLayerBias(layerIndex, deltaBias);

			// Transpose the current weight, we are "moving" backwards.
			Matrix transposedW = currentWeight.transpose();
			// Multiply the weights with the neurons of this layer
			Matrix outputMapped = transposedW.multiply(currentLayer);

			// Get the deltas by multiplying deltas with the backward-passed values.
			Matrix deltas = gradients.multiply(outputMapped.transpose());
			// Change the weights by adding the deltas.
			Matrix newWeight = currentWeight.add(deltas);

			// Set the weighs.
			this.layerConnections.setWeight(i, newWeight);

			// Update the error, which is essentially moving the errors in a backwards pass.
			// This is one of the things where I feel unsure of my understanding.
			errors = newWeight.transpose().multiply(previousErrors);
		}
	}

	/**
	 * Feed the input through the network for classification.
	 *
	 * @param in values to predict
	 * @return classified values.
	 */
	private Matrix feedForward(double[] in) {
		// Make input into matrix.
		Matrix input = Matrix.fromArray(in);

		// Assure that matrix size matches.
		if (input.getRows() != this.layerConnections.getInputNodes()) {
			throw new IllegalArgumentException("Dimensions do not match.");
		}

		// Get biases and weights from the network
		Matrix[] biases = this.layerConnections.getBiases();
		Matrix[] weights = this.layerConnections.getWeights();

		// Set the neurons of the first layer (input layer) to the input.
		this.layerConnections.setActiveLayer(0, input);

		Matrix output;
		Matrix inputIterated = input;
		for (int i = 1; i < weights.length + 1; i++) { // Loop over all weights
			// Weights multiplied with the input
			Matrix hiddenMultiplication = weights[i - 1].multiply(inputIterated);
			Matrix bias = biases[i];

			Matrix biasedHiddenMultiplication = hiddenMultiplication.add(bias);
			inputIterated = function.functionToMatrix(biasedHiddenMultiplication);
			// Set current layers neurons to the inputIterated matrix.
			this.layerConnections.setActiveLayer(i, inputIterated);
		}

		output = inputIterated;

		return output;
	}

	public void displayNetwork() {
		// TODO: Remove me
		this.layerConnections.printLayers();
		this.layerConnections.printWeights();
	}
}
