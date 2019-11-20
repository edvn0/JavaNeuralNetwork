package neuralnetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import math.ActivationFunction;
import math.ErrorFunction;
import matrix.Matrix;
import neuralnetwork.structures.LayerConnectionList;
import org.jetbrains.annotations.NotNull;

/**
 * A multi layer perceptron network.
 */
public class NeuralNetwork implements Serializable, Trainable {

	// Serial ID
	private static final long serialVersionUID = 0L;

	// Learning rate
	private double learningRate;

	// All activation functions for all layers
	private ActivationFunction[] functions;

	// The error function to minimize.
	private ErrorFunction errorFunction;

	// The structure that holds everything.
	private LayerConnectionList layerConnections; // Maps layer 0 to 1, 1 to 2, etc. Will "store" the

	private int totalLayers;

	public NeuralNetwork(double learning, ActivationFunction[] functions, ErrorFunction function,
		int[] sizes) {
		this.learningRate = learning;
		this.functions = functions;
		this.layerConnections = new LayerConnectionList(functions, sizes);
		this.errorFunction = function;
		this.totalLayers = this.layerConnections.getTotalLayers();
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void train(List<Matrix[]> training, String method) {

	}

	/**
	 * Provides an implementation of SGD for this neural network.
	 *
	 * @param training a Collections object with Matrix[] objects, Matrix[0] is the data, Matrix[1]
	 * is the label.
	 * @param test a Collections object with Matrix[] objects, Matrix[0] is the data, Matrix[1] is
	 * the label.
	 * @param epochs how many iterations are we doing SGD for
	 * @param batchSize how big is the batch size, typically 32.
	 */
	public void stochasticGradientDescent(@NotNull List<Matrix[]> training,
		@NotNull List<Matrix[]> test, int epochs,
		int batchSize) {
		int trDataSize = training.size();
		int teDataSize = test.size();
		for (int i = 0; i < epochs; i++) {
			Collections.shuffle(training);
			System.out.println("Calculating epoch: " + (i + 1) + ".");
			for (int j = 0; j < trDataSize - batchSize; j += batchSize) {
				calculateMiniBatch(training.subList(j, j + batchSize));
			}
			int correct = evaluateTestData(test);
			System.out.println("Epoch " + (i + 1) + ": " + correct + "/" + teDataSize);
		}
	}

	private int evaluateTestData(List<Matrix[]> test) {
		int correct = 0;
		for (int i = 0; i < test.size(); i++) {
			// data[i] = {data, correctLabels}
			Matrix data = test.get(i)[0];
			Matrix correctLabels = test.get(i)[1];

			Matrix fedForward = predict(data);
			int val = this.maxLabel(fedForward);
			if (correctLabels.getElement(0, 0) == val) {
				correct++;
			}

		}
		return correct;
	}

	private int maxLabel(Matrix fedForward) {
		double[] data = fedForward.toArray();
		int index = 0;
		double max = data[0];
		for (int i = 1; i < data.length; i++) {
			if (data[i] > max) {
				max = data[i];
				index = i;
			}
		}
		return index;
	}

	private void calculateMiniBatch(List<Matrix[]> subList) {
		int size = subList.size();
		Matrix[] dB = new Matrix[this.totalLayers - 1];
		Matrix[] dW = new Matrix[this.totalLayers - 1];
		for (int i = 0; i < this.totalLayers - 1; i++) {
			Matrix bias = this.layerConnections.getBias(i);
			Matrix weight = this.layerConnections.getWeight(i);
			dB[i] = new Matrix(bias.getRows(), bias.getColumns());
			dW[i] = new Matrix(weight.getRows(), weight.getColumns());
		}

		for (int i = 0; i < size; i++) {
			List<Matrix[]> deltas = backPropagate(subList.get(i)[0], subList.get(i)[1]);
			Matrix[] deltaB = deltas.get(0);
			Matrix[] deltaW = deltas.get(1);

			for (int j = 0; j < this.totalLayers - 1; j++) {
				dB[j] = dB[j].add(deltaB[j]);
				dW[j] = dW[j].add(deltaW[j]);
			}
		}

		for (int i = 0; i < this.totalLayers - 1; i++) {
			Matrix cW = this.layerConnections.getWeight(i);
			Matrix cB = this.layerConnections.getBias(i);

			Matrix scaledDeltaB = dB[i].map((e) -> e * (this.learningRate / size));
			Matrix scaledDeltaW = dW[i].map((e) -> e * (this.learningRate / size));

			this.layerConnections.setWeight(i, cW.subtract(scaledDeltaW));
			this.layerConnections.setLayerBias(i, cB.subtract(scaledDeltaB));
		}
	}

	private Matrix[] getDeltas(Matrix[] toCopyFrom) {
		Matrix[] deltas = new Matrix[toCopyFrom.length];
		for (int i = 0; i < deltas.length; i++) {
			int rows = toCopyFrom[i].getRows();
			int cols = toCopyFrom[i].getColumns();
			deltas[i] = new Matrix(rows, cols);
		}
		return deltas;
	}

	private void backPropFeedForward(Matrix starter, List<Matrix> actives, List<Matrix> vectors,
		Matrix[] weights, Matrix[] biases) {
		Matrix toPredict = starter;
		actives.add(toPredict);
		for (int i = 0; i < this.totalLayers - 1; i++) {
			Matrix x = weights[i].multiply(toPredict).add(biases[i]);
			vectors.add(x);

			toPredict = this.functions[i].applyFunction(x);
			actives.add(toPredict);
		}
	}

	private List<Matrix[]> backPropagate(Matrix toPredict, Matrix correct) {
		List<Matrix[]> totalDeltas = new ArrayList<>();

		Matrix[] weights = this.layerConnections.getWeights();
		Matrix[] biases = this.layerConnections.getBiases();

		Matrix[] deltaBiases = this.getDeltas(biases);
		Matrix[] deltaWeights = this.getDeltas(weights);

		// Perform Feed Forward here...
		List<Matrix> activations = new ArrayList<>();
		List<Matrix> xVector = new ArrayList<>();
		this.backPropFeedForward(toPredict, activations, xVector, weights, biases);
		// End feedforward

		// Calculate error signal for last layer
		Matrix error = errorFunction
			.applyErrorFunction(activations.get(activations.size() - 1), correct);
		Matrix deltaError = error
			.hadamard(functions[activations.size() - 1]
				.applyDerivative(xVector.get(xVector.size() - 1)));

		// Set the deltas to the error signals of bias and weight.
		deltaBiases[deltaBiases.length - 1] = deltaError;
		deltaWeights[deltaWeights.length - 1] = deltaError
			.multiply(activations.get(activations.size() - 2).transpose());

		// Now iteratively apply the rule
		for (int k = deltaBiases.length - 2; k >= 0; k--) {
			Matrix z = xVector.get(k);
			Matrix differentiate = functions[k].applyDerivative(z);

			deltaError = weights[k + 1].transpose().multiply(deltaError)
				.hadamard(differentiate);

			deltaBiases[k] = deltaError;
			deltaWeights[k] = deltaError.multiply(activations.get(k).transpose());
		}
		totalDeltas.add(deltaBiases);
		totalDeltas.add(deltaWeights);

		return totalDeltas;
	}

	@Override
	public Matrix predict(Matrix in) {
		return feedForward(in);
	}

	/**
	 * Feed the input through the network for classification.
	 *
	 * @param in values to predict
	 * @return classified values.
	 */
	private Matrix feedForward(Matrix in) {
		// Make input into matrix.
		Matrix input = in;
		Matrix[] weights = this.layerConnections.getWeights();
		Matrix[] biases = this.layerConnections.getBiases();
		for (int i = 0; i < this.totalLayers - 1; i++) {
			input = functions[i].applyFunction(weights[i].multiply(input).add(biases[i]));
		}

		return input;
	}

	public void displayWeights() {
		int i = 0;
		for (Matrix w : this.layerConnections.getWeights()) {
			System.out.println("Weight from " + i + " to " + (i + 1));
			System.out.println(w);
			i++;
		}
	}
}
