package neuralnetwork;

import java.io.Serializable;
import java.util.Map.Entry;
import math.ActivationFunction;
import math.SigmoidFunction;
import matrix.Matrix;
import neuralnetwork.layer.Connection;
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

	public NeuralNetwork(int inputNodes, int outputNodes) {
		this.layerConnections = new LayerConnectionList(inputNodes, 3, 30, outputNodes);
		this.learningRate = 10e-2;

		this.function = new SigmoidFunction();
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

	private void backPropagate(Matrix fedForward, Matrix correct) {
		// Calculate errors.
		final Matrix errors = correct.subtract(fedForward);

		final Matrix gradients = this.function.applyDerivativeFunctionToMatrix(fedForward);
		final Matrix mappedGradients = gradients.hadamard(errors)
			.map((e) -> (e * this.learningRate));

		Matrix newOutput = fedForward;
		for (int i = this.layerConnections.amountOfWeights() - 1; i >= 0; i--) {
			// Weights from i to i - 1., first iteration this takes the weights from
			// last to second to last. this weight is a Matrix(connection[i].outputNodes, outputNodes)
			// For a FANN with hiddenNodes = 3 for the XOR problem, we get that the last
			// weight matrix is a Matrix(3,2)

			// Change all weight values of each weight matrix using the
			// formula: weight(old) +
			// learning rate * output error * output(neurons i) * output(neurons i+1) * ( 1 - output(neurons i+1) )
			Matrix weights = this.layerConnections.getWeights(i);
			Matrix transposedWeights = weights.transpose();
			newOutput = transposedWeights.multiply(newOutput);
			Matrix bias = this.layerConnections.getLayer(i).getBias();
			newOutput = newOutput.add(bias);
			mappedGradients.show();
			Matrix deltas = newOutput.multiply(mappedGradients.transpose());
			Matrix newWeights = weights.add(deltas.transpose());
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
		for (int i = 0; i < this.layerConnections.amountOfWeights(); i++) {
			// Calculate the matrix multiplication + bias.
			outputMatrix = this.layerConnections.calculateLayer(outputMatrix, i, i + 1);
			// Activate the neuron.
			outputMatrix = function.applyFunctionToMatrix(outputMatrix);

		}
		return outputMatrix;
	}

	public void displayNetwork() {
		for (Entry<Connection, Matrix> entry : this.layerConnections.getWeightsEntries()) {
			System.out.println(
				"Connection:" + entry.getKey().getFrom().getIndexInNetwork() + " to: " + entry
					.getKey().getTo().getIndexInNetwork());
			entry.getValue().show();
			System.out.println();
		}
	}
}
