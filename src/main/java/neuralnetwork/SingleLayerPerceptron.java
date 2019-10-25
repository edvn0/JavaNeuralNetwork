package neuralnetwork;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map.Entry;
import math.ActivationFunction;
import math.ActivationFunctionFactory;
import matrix.Matrix;

public class SingleLayerPerceptron implements Serializable, Trainable {

	private static final long serialVersionUID = 1233L;

	private double learningRate;
	private Matrix inputHiddenWeights;
	private Matrix outputHiddenWeights;
	private HashMap<Integer, Matrix> layers;

	private ActivationFunctionFactory functionFactory;
	private ActivationFunction function;

	private int inputNodes, outputNodes, numOfLayers;

	private Matrix hiddenBias, outputBias;

	public SingleLayerPerceptron(int inputNodes, int hiddenNodes, int outputNodes,
		double learningRate) {
		this.inputHiddenWeights = Matrix.random(hiddenNodes, inputNodes);
		this.outputHiddenWeights = Matrix.random(outputNodes, hiddenNodes);

		this.hiddenBias = Matrix.random(hiddenNodes, 1);
		this.outputBias = Matrix.random(outputNodes, 1);

		this.learningRate = learningRate;

		this.functionFactory = new ActivationFunctionFactory();
	}

	@Deprecated
	public SingleLayerPerceptron(int inputNodes, int outputNodes, int layers,
		int... numOfNodesInLayers) {
		this.inputNodes = inputNodes;
		this.outputNodes = outputNodes;
		this.numOfLayers = layers;
		this.layers = new HashMap<>();
		for (int i = 0; i < layers; i++) {
			Matrix networkLayer = new Matrix(numOfNodesInLayers[i], 1);
			this.layers.put(i + 1, networkLayer);
		}
	}


	/**
	 * Feeds the input forward in the neural network. Takes a double[] of size INPUT_NODES.
	 *
	 * @param input double[] with values to be predicted.
	 * @return a Matrix(actually a vector, k*1 Matrix) with the predicted outputs.
	 */
	public Matrix predict(double[] input) {
		Matrix inputMatrix = Matrix.fromArray(input);
		Matrix hidden = this.inputHiddenWeights.multiply(inputMatrix);

		hidden = hidden.add(this.hiddenBias);
		hidden = this.function.applyFunctionToMatrix(hidden);

		Matrix output = this.outputHiddenWeights.multiply(hidden);
		output = output.add(this.outputBias);
		output = this.function.applyFunctionToMatrix(output);

		return output;
	}

	public void train(double[] inputs, double[] knownResults) {
		Matrix inputMatrix = Matrix.fromArray(inputs);
		Matrix targetMatrix = Matrix.fromArray(knownResults);

		//----------
		// Calculate feedforward with inputs.
		//----------

		// From input layer -> hidden layer.
		Matrix hidden = this.inputHiddenWeights.multiply(inputMatrix);
		hidden = hidden.add(this.hiddenBias);
		hidden = this.function.applyFunctionToMatrix(hidden);

		// From hidden layer -> output layer. ActivationFunction(Weighted sum (hidden) + bias).
		Matrix outputs = this.outputHiddenWeights.multiply(hidden);
		outputs = outputs.add(this.outputBias);
		outputs = this.function.applyFunctionToMatrix(outputs);

		// How incorrect was this prediction?
		Matrix outputErrors = targetMatrix.subtract(outputs);

		// Calculate gradients, i.e. Activation derivatives of all elements.
		Matrix gradients = this.function.applyDerivativeFunctionToMatrix(outputs);
		gradients = gradients.hadamard(outputErrors);
		gradients = gradients.map((e) -> e * learningRate);

		Matrix hiddenTransposed = hidden.transpose();
		Matrix hiddenOutputDeltas = gradients.multiply(hiddenTransposed);

		this.outputHiddenWeights = this.outputHiddenWeights.add(hiddenOutputDeltas);
		this.outputBias = this.outputBias.add(gradients);

		Matrix outputHiddenWeightsTransposed = this.outputHiddenWeights.transpose();
		Matrix hiddenErrors = outputHiddenWeightsTransposed.multiply(outputErrors);

		Matrix hiddenGradient = this.function.applyDerivativeFunctionToMatrix(hidden);
		hiddenGradient = hiddenGradient.hadamard(hiddenErrors);
		hiddenGradient = hiddenGradient.map((e) -> e * learningRate);

		Matrix inputsTransposed = inputMatrix.transpose();
		Matrix inputHiddenWeightDeltas = hiddenGradient.multiply(inputsTransposed);

		this.inputHiddenWeights = this.inputHiddenWeights.add(inputHiddenWeightDeltas);
		this.hiddenBias = this.hiddenBias.add(hiddenGradient);
	}

	public void setActivationFunction(String function) {
		this.function = new ActivationFunctionFactory().getActivationFunctionByKey(function);
	}


	public void setDefaultValues() {
		this.learningRate = 0.1;
		this.function = functionFactory.getActivationFunctionByKey("SIGMOID");
	}

	public void printNeuralNetwork() {
		System.out.println("Input nodes: " + this.inputNodes);
		for (Entry<Integer, Matrix> entry : this.layers.entrySet()) {
			System.out.println("Layer: " + entry.getKey());
			entry.getValue().show();
		}
		System.out.println("Output nodes: " + this.outputNodes);
	}
}
