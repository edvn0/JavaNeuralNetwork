package neuralnetwork;

import java.io.Serializable;
import java.util.List;
import math.ActivationFunction;
import math.ReluFunction;
import math.TanhFunction;
import matrix.Matrix;

public class SingleLayerPerceptron implements Serializable, Trainable {

	private static final long serialVersionUID = 1233L;

	private double learningRate;
	private Matrix inputToHiddenWeights;
	private Matrix hiddenToOutputWeights;

	private transient ActivationFunction firstLayerFunction;
	private transient ActivationFunction lastLayerFunction;

	private int inputNodes, outputNodes;

	private Matrix hiddenBias, outputBias;

	public SingleLayerPerceptron(int inputNodes, int hiddenNodes, int outputNodes,
		double learningRate) {
		this.inputToHiddenWeights = Matrix.random(hiddenNodes, inputNodes);
		this.hiddenToOutputWeights = Matrix.random(outputNodes, hiddenNodes);

		this.hiddenBias = Matrix.random(hiddenNodes, 1);
		this.outputBias = Matrix.random(outputNodes, 1);

		this.firstLayerFunction = new ReluFunction();
		this.lastLayerFunction = new TanhFunction();

		this.learningRate = learningRate;
	}


	/**
	 * Feeds the input forward in the neural network. Takes a double[] of size INPUT_NODES.
	 *
	 * @param input double[] with values to be predicted.
	 * @return a Matrix(actually a vector, k*1 Matrix) with the predicted outputs.
	 */
	public Matrix predict(Matrix input) {
		Matrix inputMatrix = input;
		Matrix hidden = this.inputToHiddenWeights.multiply(inputMatrix);

		hidden = hidden.add(this.hiddenBias);
		hidden = this.firstLayerFunction.applyFunction(hidden);

		Matrix output = this.hiddenToOutputWeights.multiply(hidden);
		output = output.add(this.outputBias);
		output = this.lastLayerFunction.applyFunction(output);

		return output;
	}

	public void train(List<Matrix[]> testData, String method) {
		//----------
		// Calculate feedforward with inputs.
		//----------

		for (int i = 0; i < testData.size(); i++) {
			Matrix inputs = testData.get(i)[0];
			Matrix knownResults = testData.get(i)[1];
			// From input layer -> hidden layer.
			Matrix hidden = this.inputToHiddenWeights.multiply(inputs);
			hidden = hidden.add(this.hiddenBias);
			hidden = this.firstLayerFunction.applyFunction(hidden);

			// From hidden layer -> output layer. ActivationFunction(Weighted sum (hidden) + bias).
			Matrix outputs = this.hiddenToOutputWeights.multiply(hidden);
			outputs = outputs.add(this.outputBias);
			outputs = this.lastLayerFunction.applyFunction(outputs);

			// How incorrect was this prediction?
			Matrix outputErrors = knownResults.subtract(outputs);

			// Calculate gradients, i.e. Activation derivatives of all elements.
			// FIXME: This was changed from softmax, might change back
			Matrix gradients = this.lastLayerFunction.applyDerivative(outputs);
			gradients = gradients.hadamard(outputErrors);
			gradients = gradients.map((e) -> e * learningRate);

			Matrix hiddenTransposed = hidden.transpose();
			Matrix hiddenOutputDeltas = gradients.multiply(hiddenTransposed);

			this.hiddenToOutputWeights = this.hiddenToOutputWeights.add(hiddenOutputDeltas);
			this.outputBias = this.outputBias.add(gradients);

			Matrix outputHiddenWeightsTransposed = this.hiddenToOutputWeights.transpose();
			Matrix hiddenErrors = outputHiddenWeightsTransposed.multiply(outputErrors);

			Matrix hiddenGradient = this.firstLayerFunction.applyDerivative(hidden);
			hiddenGradient = hiddenGradient.hadamard(hiddenErrors);
			hiddenGradient = hiddenGradient.map((e) -> e * learningRate);

			Matrix inputsTransposed = inputs.transpose();
			Matrix inputHiddenWeightDeltas = hiddenGradient.multiply(inputsTransposed);

			this.inputToHiddenWeights = this.inputToHiddenWeights.add(inputHiddenWeightDeltas);
			this.hiddenBias = this.hiddenBias.add(hiddenGradient);
		}
	}
}
