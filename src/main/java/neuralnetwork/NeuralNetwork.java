package neuralnetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import math.ActivationFunction;
import math.ErrorFunction;
import math.ReluFunction;
import math.TanhFunction;
import matrix.Matrix;
import neuralnetwork.structures.Layer;
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

	// Keeps track of previously activated neurons in the network.
	private Layer[] activatedLayers;

	private int totalLayers;

	public NeuralNetwork(double learning, ActivationFunction[] functions, ErrorFunction function,
		int[] sizes) {
		this.learningRate = learning;
		this.functions = functions;
		this.layerConnections = new LayerConnectionList(functions, sizes);
		this.activatedLayers = this.layerConnections.getLayers();
		this.errorFunction = function;
	}

	/**
	 * Initialises the functions to tanh for all hidden and softmax for last.
	 */
	public void initialiseFunctions() {
		for (int i = 0; i < this.functions.length - 1; i++) {
			this.functions[i] = new ReluFunction();
		}
		functions[functions.length - 1] = new TanhFunction();
	}


	/**
	 * To test the neural network, we want to be able to analyse its matrix multiplication
	 * correctness.
	 *
	 * @param activatedLayers all the layers activations
	 * @param weights weights from all layers
	 * @param biases the biases
	 */
	public NeuralNetwork(double[][] activatedLayers, Matrix[] weights,
		Matrix[] biases) {
		this.learningRate = 0.1d;
		this.functions = new ActivationFunction[activatedLayers.length];
		initialiseFunctions();
		layerConnections = new LayerConnectionList(activatedLayers, weights, biases);
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

			System.out.println(fedForward);

			int maxResultRow = 0;
			int maxOutputRow = 0;

			for (int j = 0; j < correctLabels.getRows(); j++) {
				if (correctLabels.getElement(j, 0) > correctLabels.getElement(maxResultRow, 0)) {
					maxResultRow = j;
				}
				if (fedForward.getElement(j, 0) > fedForward.getElement(maxOutputRow, 0)) {
					maxOutputRow = j;
				}
			}
			if (maxOutputRow == maxResultRow) {
				correct++;
			}
		}
		return correct;
	}

	private void calculateMiniBatch(List<Matrix[]> subList) {
		int size = subList.size();
		Matrix[] dB = new Matrix[this.layerConnections.getTotalLayers() - 1];
		Matrix[] dW = new Matrix[this.layerConnections.getTotalLayers() - 1];
		for (int i = 0; i < this.layerConnections.getTotalLayers() - 1; i++) {
			Matrix bias = this.layerConnections.getBias(i);
			Matrix weight = this.layerConnections.getWeight(i);
			dB[i] = new Matrix(bias.getRows(), bias.getColumns());
			dW[i] = new Matrix(weight.getRows(), weight.getColumns());
		}

		for (int i = 0; i < size; i++) {
			List<Matrix[]> deltas = backPropagate(subList.get(i)[0], subList.get(i)[1]);
			Matrix[] deltaB = deltas.get(0);
			Matrix[] deltaW = deltas.get(1);

			for (int j = 0; j < this.layerConnections.getTotalLayers() - 1; j++) {
				dB[j] = dB[j].add(deltaB[j]);
				dW[j] = dW[j].add(deltaW[j]);
			}
		}

		for (int i = 0; i < this.layerConnections.getTotalLayers() - 1; i++) {
			Matrix cW = this.layerConnections.getWeight(i);
			Matrix cB = this.layerConnections.getBias(i);

			Matrix scaledDeltaB = dB[i].map((e) -> e * (this.learningRate / size));
			Matrix scaledDeltaW = dW[i].map((e) -> e * (this.learningRate / size));

			this.layerConnections.setWeight(i, cW.subtract(scaledDeltaW));
			this.layerConnections.setLayerBias(i, cB.subtract(scaledDeltaB));
		}
	}

	private List<Matrix[]> backPropagate(Matrix toPredict, Matrix correct) {
		Matrix[] deltaBiases = new Matrix[this.layerConnections.getTotalLayers() - 1];
		Matrix[] deltaWeights = new Matrix[this.layerConnections.getTotalLayers() - 1];

		for (int i = 0; i < deltaBiases.length; i++) {
			Matrix cBias, cWeights;
			cBias = this.layerConnections.getLayer(i).getBias();
			deltaBiases[i] = new Matrix(cBias.getRows(), cBias.getColumns());
			cWeights = this.layerConnections.getWeight(i);
			deltaWeights[i] = new Matrix(cWeights.getRows(), cWeights.getColumns());
		}

		// Perform Feed Forward here...
		Matrix[] weights = this.layerConnections.getWeights();
		Matrix[] biases = this.layerConnections.getBiases();
		// add activated layers and neurons.
		List<Matrix> activations = new ArrayList<>();
		Matrix activation = toPredict;
		activations.add(activation);

		List<Matrix> xVector = new ArrayList<>();
		for (int i = 0; i < this.layerConnections.getTotalLayers() - 1; i++) {
			// out = wX+B
			Matrix x = weights[i].multiply(activation).add(biases[i]);
			xVector.add(x);

			// activated = f(out)
			activation = functions[i].applyFunction(x);
			activations.add(activation);
		}
		// End feedforward

		Matrix error = errorFunction
			.applyFunction(activations.get(activations.size() - 1), correct);
		Matrix deltaError = error
			.hadamard(functions[activations.size() - 1]
				.applyDerivative(xVector.get(xVector.size() - 1)));

		deltaBiases[deltaBiases.length - 1] = deltaError;
		deltaWeights[deltaWeights.length - 1] = deltaError
			.multiply(activations.get(activations.size() - 2).transpose());

		for (int k = deltaBiases.length - 2; k >= 0; k--) {
			Matrix z = xVector.get(k);
			Matrix differentiate = functions[k].applyDerivative(z);

			deltaError = this.layerConnections.getWeight(k + 1).transpose().multiply(deltaError)
				.hadamard(differentiate);

			deltaBiases[k] = deltaError;
			deltaWeights[k] = deltaError.multiply(activations.get(k).transpose());
		}
		List<Matrix[]> totalDeltas = new ArrayList<>();
		totalDeltas.add(deltaBiases);
		totalDeltas.add(deltaWeights);

		return totalDeltas;
		/*
		Layer last = this.activatedLayers[finalLayer];
		Layer secondToLast = this.activatedLayers[finalLayer - 1];
		Matrix lastWeight = this.layerConnections.getWeight(finalWeight);


		//sigma_L = (x_L - target) o (f_L'(w_(L-1)*a_(L-1))


		// w_(L-1)*a_(L-1) = Wa
		Matrix inner = lastWeight.multiply(secondToLast.getValues());

		// f_L'(Wa)
		Matrix derive = functions[finalLayer].applyDerivative(inner);

		// calculate sigma_L and then scale by learning rate.
		Matrix sigmaLast = error.hadamard(derive);
		Matrix sigmaLastScaled = sigmaLast.map((e) -> e * this.learningRate);

		// new W_i = W_i - (lr o dEdW) <=>
		// new W_i = W_i - (lR o sigma_i*x_(i-1)^T)
		Matrix dEdW = sigmaLast.multiply(secondToLast.getValues().transpose());
		dEdW = dEdW.map(e -> e * this.learningRate);

		// new W_i
		Matrix updatedWeight = lastWeight.subtract(dEdW);

		// new B_i = B_i - lr o sigma_i
		Matrix updatedBias = last.getBias().subtract(sigmaLastScaled);

		// set new values in NN
		last.calculateBias(updatedBias);
		this.layerConnections.setWeight(finalWeight, updatedWeight);

		Matrix iterate = sigmaLast;
		for (int i = finalLayer - 1; i >= 0; i--) {
			if (i == 0) {

			} else {
				int w = i - 1;
				// sigma_i = W_(i+1)^T*sigma_(i+1) had f'(W_(i)*Layer_(i-1)^T))
				// need w_(i+1), last sigma, w_(i) and layer_(i-1). We are modifying layer i
				Matrix pW = this.layerConnections.getWeight(w + 1);
				Matrix cW = this.layerConnections.getWeight(w);
				Layer cL = this.activatedLayers[i]; // alter bias for...
				Layer nL = this.activatedLayers[i - 1];

				// CALC SIGMA
				Matrix derivative = this.functions[i].applyDerivative(cW.multiply(nL.getValues()));
				Matrix sigmaWeight = pW.transpose().multiply(iterate);
				Matrix sigmaI = sigmaWeight.hadamard(derivative);

				Matrix sigmaScaledI = sigmaI.map(e -> e * this.learningRate);
				// END CALC SIGMA

				Matrix weightDelta = sigmaScaledI.multiply(nL.getValues().transpose());

				Matrix updatedWeights = cW.subtract(weightDelta);

				this.layerConnections.setActiveLayerBias(i,
					cL.getBias().subtract(sigmaScaledI));

				this.layerConnections.setWeight((w), updatedWeights);
				iterate = sigmaI;
			}
		}

		// Reset layers to zero after iteration.
		this.layerConnections.resetLayers();
		this.activatedLayers = this.layerConnections.getLayers();
		*/
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
		// Assure that matrix size matches.
		for (int i = 0; i < this.layerConnections.getTotalLayers() - 1; i++) {
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

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("NeuralNetwork{");
		sb.append("activatedLayers=").append(Arrays.deepToString(activatedLayers));
		sb.append(", functions=").append(Arrays.toString(functions));
		sb.append(", learningRate=").append(learningRate);
		sb.append('}');
		return sb.toString();
	}
}
