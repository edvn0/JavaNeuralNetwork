package neuralnetwork;

import java.io.Serializable;
import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SoftmaxFunction;
import math.activations.TanhFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.errors.MeanSquaredErrorFunction;
import matrix.Matrix;
import org.ujmp.core.DenseMatrix;
import utilities.MatrixUtilities;

public class SingleLayerPerceptron implements Serializable, Trainable {

	private static final long serialVersionUID = 1233L;

	private double learningRate;
	private DenseMatrix inputToHiddenWeights;
	private DenseMatrix hiddenToOutputWeights;

	private ActivationFunction firstLayerFunction;
	private ActivationFunction lastLayerFunction;

	private ErrorFunction err;

	private DenseMatrix hiddenBias, outputBias;

	public SingleLayerPerceptron(int inputNodes, int hiddenNodes, int outputNodes,
		double learningRate) {
		this.inputToHiddenWeights = org.ujmp.core.Matrix.Factory.randn(hiddenNodes, inputNodes);
		this.hiddenToOutputWeights = org.ujmp.core.Matrix.Factory.randn(outputNodes, hiddenNodes);

		this.hiddenBias = org.ujmp.core.Matrix.Factory.randn(hiddenNodes, 1);
		this.outputBias = org.ujmp.core.Matrix.Factory.randn(outputNodes, 1);

		this.firstLayerFunction = new ReluFunction();

		if (outputNodes > 1) {
			this.lastLayerFunction = new SoftmaxFunction();
			this.err = new CrossEntropyErrorFunction();
		} else {
			this.lastLayerFunction = new TanhFunction();
			this.err = new MeanSquaredErrorFunction();
		}

		this.learningRate = learningRate;
	}


	/**
	 * Feeds the input forward in the neural network. Takes a double[] of size INPUT_NODES.
	 *
	 * @param input double[] with values to be predicted.
	 *
	 * @return a Matrix(actually a vector, k*1 Matrix) with the predicted outputs.
	 */
	public DenseMatrix predict(DenseMatrix input) {
		DenseMatrix hidden = (DenseMatrix) this.inputToHiddenWeights.mtimes(input);

		hidden = (DenseMatrix) hidden.plus(this.hiddenBias);
		hidden = this.firstLayerFunction.applyFunction(hidden);

		DenseMatrix output = (DenseMatrix) this.hiddenToOutputWeights.mtimes(hidden);
		output = (DenseMatrix) output.plus(this.outputBias);
		output = this.lastLayerFunction.applyFunction(output);

		return output;
	}

	/**
	 * Train the network with given inputs testData and a correct label.
	 *
	 * @param testData a {@link Matrix} object
	 * @param correct  labels for the data.
	 */
	public void train(DenseMatrix testData, DenseMatrix correct) {
		//----------
		// Calculate feedforward with inputs.
		//----------
		// From input layer -> hidden layer.
		DenseMatrix hidden = (DenseMatrix) this.inputToHiddenWeights.mtimes(testData);
		hidden = (DenseMatrix) hidden.plus(this.hiddenBias);
		hidden = this.firstLayerFunction.applyFunction(hidden);

		// From hidden layer -> output layer. ActivationFunction(Weighted sum (hidden) + bias).
		DenseMatrix outputs = (DenseMatrix) this.hiddenToOutputWeights.mtimes(hidden);
		outputs = (DenseMatrix) outputs.plus(this.outputBias);
		outputs = this.lastLayerFunction.applyFunction(outputs);

		// How incorrect was this prediction?
		DenseMatrix outputErrors = err.applyErrorFunction(outputs, correct);

		// Calculate gradients, i.e. Activation derivatives of all elements.
		// FIXME: This was changed from softmax, might change back
		DenseMatrix gradients = this.lastLayerFunction.applyDerivative(outputs);
		gradients = (DenseMatrix) gradients.times(outputErrors);
		gradients = MatrixUtilities.map(gradients, (e) -> e * this.learningRate);

		DenseMatrix hiddenTransposed = (DenseMatrix) hidden.transpose();
		DenseMatrix hiddenOutputDeltas = (DenseMatrix) gradients.mtimes(hiddenTransposed);

		this.hiddenToOutputWeights = (DenseMatrix) this.hiddenToOutputWeights
			.plus(hiddenOutputDeltas);
		this.outputBias = (DenseMatrix) this.outputBias.plus(gradients);

		DenseMatrix outputHiddenWeightsTransposed = (DenseMatrix) this.hiddenToOutputWeights
			.transpose();
		DenseMatrix hiddenErrors = (DenseMatrix) outputHiddenWeightsTransposed.mtimes(outputErrors);

		DenseMatrix hiddenGradient = this.firstLayerFunction.applyDerivative(hidden);
		hiddenGradient = (DenseMatrix) hiddenGradient.times(hiddenErrors);
		hiddenGradient = MatrixUtilities.map(hiddenGradient, (e) -> e * learningRate);

		DenseMatrix inputsTransposed = (DenseMatrix) testData.transpose();
		DenseMatrix inputHiddenWeightDeltas = (DenseMatrix) hiddenGradient.mtimes(inputsTransposed);

		this.inputToHiddenWeights = (DenseMatrix) this.inputToHiddenWeights
			.plus(inputHiddenWeightDeltas);
		this.hiddenBias = (DenseMatrix) this.hiddenBias.plus(hiddenGradient);

	}
}
