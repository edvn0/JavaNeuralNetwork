package neuralnetwork;

import java.io.Serializable;
import java.util.Map.Entry;
import math.ActivationFunction;
import math.SigmoidFunction;
import matrix.Matrix;
import neuralnetwork.layer.Connection;
import neuralnetwork.layer.Layer;
import neuralnetwork.layer.LayerConnectionList;
import utilites.NeuralNetworkOptions;

public class NeuralNetwork implements Serializable, Trainable {

	private NeuralNetworkOptions options;

	private double learningRate;
	private int totalLayers;

	private ActivationFunction function;

	private LayerConnectionList layerConnections; // Maps layer 0 to 1, 1 to 2, etc. Will "store" the

	private NeuralNetwork(NeuralNetworkOptions options) {
		this.options = options;
	}

	public NeuralNetwork(int inputNodes, int hiddenLayers, int nodesInHiddenLayers,
		int outputNodes, double learningRate) {
		this.layerConnections = new LayerConnectionList(inputNodes, hiddenLayers,
			nodesInHiddenLayers, outputNodes);
		this.learningRate = learningRate;
		this.totalLayers = 1 + 1 + hiddenLayers;

		this.function = new SigmoidFunction();
	}

	public NeuralNetwork(int inputNodes, int outputNodes) {
		this.layerConnections = new LayerConnectionList(inputNodes, 1, 30, outputNodes);
		this.learningRate = 10e-2;
		this.totalLayers = 2 + 1;
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

	private void backPropagate(Matrix in, Matrix correct) {
		Matrix errors = correct.subtract(in);

		Matrix gradients = this.function.applyDerivativeFunctionToMatrix(in);
		gradients = gradients.hadamard(errors).map((e) -> e * learningRate);

		for (int i = this.layerConnections.amountWeightMatrices() - 1; i >= 0; i--) {
			Matrix weights = this.layerConnections.getWeights(i);
			Matrix deltas = gradients.multiply(weights);
			Matrix newWeights = weights.add(deltas);
			newWeights.show();
			this.layerConnections.setWeights(i, newWeights);
		}

	}

	private Matrix feedForward(double[] in) {
		// Make input into matrix.
		Matrix input = Matrix.fromArray(in);
		if (input.getRows() != this.layerConnections.getInputNodes()) {
			throw new IllegalArgumentException("Dimensions do not match.");
		}

		Matrix outputMatrix = input;
		for (int i = 0; i < this.layerConnections.amountWeightMatrices(); i++) {
			outputMatrix = this.layerConnections.calculateLayer(outputMatrix, i, i + 1);
			outputMatrix = function.applyFunctionToMatrix(outputMatrix);

		}
		return outputMatrix;
	}

	public void displayNetwork() {
		for (Entry<Connection, Matrix> entry : this.layerConnections.getWeightsEntries()) {
			System.out
				.println("From: " + entry.getKey().getFrom() + ", To: " + entry.getKey().getTo());
			entry.getValue().show();
			System.out.println();
		}
	}
}
