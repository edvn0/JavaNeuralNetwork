package neuralnetwork;

import errors.BackpropagationError;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import math.activations.ActivationFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import org.jetbrains.annotations.NotNull;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import utilities.MatrixUtilities;

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

	// The function to evaluate the data set.
	private EvaluationFunction evaluationFunction;

	// 0 based connections, i.e., connection 0 is from Layer 0 to Layer 1.
	private DenseMatrix[] weights;

	// 0 based layering, i.e. index 0 in layers is layer 0.
	private DenseMatrix[] biases;

	// Helper field to hold the total amount of layers
	private int totalLayers;

	// Current best score for this network, used for serialisation
	private double score;

	private static transient final ArrayList<Double> xValues = new ArrayList<>();
	private static transient final ArrayList<Double> lossValues = new ArrayList<>();
	private static transient final ArrayList<Double> correctValues = new ArrayList<>();

	/**
	 * This is a wrapper constructor to facilitate the serialization concept of score.
	 *
	 * @param score a double representing the networks score
	 */
	public NeuralNetwork(double learning, ActivationFunction[] functions, ErrorFunction function,
		EvaluationFunction eval,
		int[] sizes, double score) {
		this(learning, functions, function, eval, sizes);
		this.score = score;

	}

	/**
	 * Create a Neural Network with a learning rate, all the activation functions for all layers,
	 * the error function and the function to evaluate the network, and also the sizes of the
	 * layers, for example:
	 *
	 * int[] sizes = {3,4,4,1} is a 4-layered fully connected network with 3 input nodes, 1 output
	 * nodes, 2 hidden layers with 4 nodes in each of them.
	 *
	 * @param learning  a double representing step size in back propagation.
	 * @param functions the activation functions for all layers
	 * @param function  the error function to calculate error of last layers
	 * @param eval      the evaluation function to compare the network to the data's labels
	 * @param sizes     the table to initialize layers and weights.
	 */
	public NeuralNetwork(double learning, ActivationFunction[] functions, ErrorFunction function,
		EvaluationFunction eval,
		int[] sizes) {
		this.learningRate = learning;
		this.functions = functions;
		this.errorFunction = function;
		this.totalLayers = sizes.length;
		this.evaluationFunction = eval;
		this.score = 0;

		createLayers(sizes);
		initialiseWeights(sizes);

		if ((!(functions[functions.length - 1] instanceof SoftmaxFunction)
			&& function instanceof CrossEntropyErrorFunction)) {
			throw new BackpropagationError(
				"To properly function, back-propagation needs the activation function of the last "
					+ "layer to be differentiable with respect to the error function.");
		}
	}

	private void initialiseWeights(int[] sizes) {
		this.weights = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.weights[i] = MatrixUtilities
				.map(Matrix.Factory.rand(sizes[i + 1], sizes[i]), (e) -> (2 * e - 1) * 0.1);
		}
	}

	private void createLayers(int[] sizes) {
		this.biases = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.biases[i] = MatrixUtilities
				.map(Matrix.Factory.rand(sizes[i + 1], 1), (e) -> (2 * e - 1) * 0.1);
		}
	}

	private int getTotalLayers() {
		return this.totalLayers;
	}

	public static NeuralNetwork readObject(String path) throws IOException {
		NeuralNetwork network = null;
		File file;
		path = (path.endsWith(".ser") ? path : path + ".ser");

		try {
			FileInputStream fs = new FileInputStream(file = new File(path));
			ObjectInputStream os = new ObjectInputStream(fs);

			network = (NeuralNetwork) os.readObject();

			os.close();
			fs.close();

			System.out.println("Completed deserialization, see file: " + file.getPath());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		if (null != network) {
			return network;
		} else {
			throw new IOException("Something bad happened during deserialization.");
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void train(DenseMatrix training, DenseMatrix correct) {
		calculateMiniBatch(Collections.singletonList(new NetworkInput(training, correct)));
	}

	private void calculateMiniBatch(List<NetworkInput> subList) {
		int size = subList.size();

		double scaleFactor = this.learningRate / size;

		DenseMatrix[] dB = new DenseMatrix[this.totalLayers - 1];
		DenseMatrix[] dW = new DenseMatrix[this.totalLayers - 1];
		for (int i = 0; i < this.totalLayers - 1; i++) {
			DenseMatrix bias = getBias(i);
			DenseMatrix weight = getWeight(i);
			dB[i] = Matrix.Factory.zeros(bias.getRowCount(), bias.getColumnCount());
			dW[i] = Matrix.Factory
				.zeros(weight.getRowCount(), weight.getColumnCount());
		}

		for (NetworkInput data : subList) {
			DenseMatrix dataIn = data.getData();
			DenseMatrix label = data.getLabel();
			List<DenseMatrix[]> deltas = backPropagate(dataIn, label);
			DenseMatrix[] deltaB = deltas.get(0);
			DenseMatrix[] deltaW = deltas.get(1);

			for (int j = 0; j < this.totalLayers - 1; j++) {
				dB[j] = (DenseMatrix) dB[j].plus(deltaB[j]);
				dW[j] = (DenseMatrix) dW[j].plus(deltaW[j]);
			}
		}

		for (int i = 0; i < this.totalLayers - 1; i++) {
			DenseMatrix cW = getWeight(i);
			DenseMatrix cB = getBias(i);

			DenseMatrix scaledDeltaB = (DenseMatrix) dB[i].times(scaleFactor);
			DenseMatrix scaledDeltaW = (DenseMatrix) dW[i].times(scaleFactor);

			DenseMatrix nW = (DenseMatrix) cW.minus(scaledDeltaW);
			DenseMatrix nB = (DenseMatrix) cB.minus(scaledDeltaB);

			setWeight(i, nW);
			setLayerBias(i, nB);
		}

	}

	private List<DenseMatrix[]> backPropagate(DenseMatrix toPredict, DenseMatrix correct) {
		List<DenseMatrix[]> totalDeltas = new ArrayList<>();

		DenseMatrix[] weights = getWeights();
		DenseMatrix[] biases = getBiasesAsMatrices();

		DenseMatrix[] deltaBiases = this.initializeDeltas(biases);
		DenseMatrix[] deltaWeights = this.initializeDeltas(weights);

		// Perform Feed Forward here...
		List<DenseMatrix> activations = new ArrayList<>();
		List<DenseMatrix> xVector = new ArrayList<>();

		// FIXME: Here things become NaN
		// Alters all arrays and lists.
		this.backPropFeedForward(toPredict, activations, xVector, weights, biases);
		// End feedforward

		// Calculate error signal for last layer
		DenseMatrix error;
		DenseMatrix deltaError;

		// Applies the error function to the last layer, create
		deltaError = errorFunction
			.applyErrorFunctionGradient(activations.get(activations.size() - 1), correct);

		// Set the deltas to the error signals of bias and weight.
		deltaBiases[deltaBiases.length - 1] = deltaError;
		deltaWeights[deltaWeights.length - 1] = (DenseMatrix) deltaError
			.mtimes(activations.get(activations.size() - 2).transpose());

		// Now iteratively apply the rule
		for (int k = deltaBiases.length - 2; k >= 0; k--) {
			DenseMatrix z = xVector.get(k);
			DenseMatrix differentiate = functions[k + 1].applyDerivative(z);

			deltaError = (DenseMatrix) weights[k + 1].transpose().mtimes(deltaError)
				.times(differentiate);

			deltaBiases[k] = deltaError;
			deltaWeights[k] = (DenseMatrix) deltaError.mtimes(activations.get(k).transpose());
		}

		totalDeltas.add(deltaBiases);
		totalDeltas.add(deltaWeights);

		return totalDeltas;
	}

	private double averageSum(final DenseMatrix[] in) {
		double sum = 0;
		for (DenseMatrix m : in) {
			sum += m.getValueSum();
		}
		return sum;
	}

	private DenseMatrix[] initializeDeltas(DenseMatrix[] toCopyFrom) {
		DenseMatrix[] deltas = new DenseMatrix[toCopyFrom.length];
		for (int i = 0; i < deltas.length; i++) {
			int rows = (int) toCopyFrom[i].getRowCount();
			int cols = (int) toCopyFrom[i].getColumnCount();
			deltas[i] = Matrix.Factory.zeros(rows, cols);
		}
		return deltas;
	}

	private void backPropFeedForward(DenseMatrix starter, List<DenseMatrix> actives,
		List<DenseMatrix> vectors,
		DenseMatrix[] weights, DenseMatrix[] biases) {
		DenseMatrix toPredict = starter;
		//actives.add(toPredict);
		actives.add(Matrix.Factory.zeros(starter.getRowCount(), starter.getColumnCount()));
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			DenseMatrix x = (DenseMatrix) weights[i].mtimes(toPredict).plus(biases[i]);
			vectors.add(x);

			toPredict = this.functions[i + 1].applyFunction(x);
			actives.add(toPredict);
		}
	}

	//-------------------------
	// Mutators
	//-------------------------
	private DenseMatrix[] getWeights() {
		return this.weights;
	}

	private DenseMatrix[] getBiasesAsMatrices() {
		DenseMatrix[] biases = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			biases[i] = getBias(i);
		}
		return biases;
	}

	private void setWeight(int i, DenseMatrix newWeights) {
		this.weights[i] = newWeights;
	}

	private DenseMatrix getWeight(int i) {
		return this.weights[i];
	}

	private DenseMatrix getBias(int i) {
		return this.biases[i];
	}

	private void setLayerBias(int i, DenseMatrix outputMatrix) {
		this.biases[i] = outputMatrix;
	}

	@Override
	public DenseMatrix predict(DenseMatrix in) {
		return feedForward(in);
	}

	/**
	 * Feed the input through the network for classification.
	 *
	 * @param in values to predict
	 *
	 * @return classified values.
	 */
	private DenseMatrix feedForward(DenseMatrix in) {
		// Make input into matrix.
		DenseMatrix input = in;
		DenseMatrix[] weights = getWeights();
		DenseMatrix[] biases = getBiasesAsMatrices();
		for (int i = 0; i < this.totalLayers - 1; i++) {
			input = functions[i + 1]
				.applyFunction((DenseMatrix) weights[i].mtimes(input).plus(biases[i]));
		}

		return input;
	}

	/**
	 * Feed the input through the network for classification.
	 *
	 * @param in values to predict
	 *
	 * @return classified values.
	 */
	private DenseMatrix feedForwardVerbose(DenseMatrix in) {
		// Make input into matrix.
		DenseMatrix input = in;

		System.out.println("Input matrix: " + Arrays.deepToString(in.toDoubleArray()));

		DenseMatrix[] weights = getWeights();
		DenseMatrix[] biases = getBiasesAsMatrices();
		for (int i = 0; i < this.totalLayers - 1; i++) {
			input = functions[i + 1]
				.applyFunction((DenseMatrix) weights[i].mtimes(input).plus(biases[i]));
			System.out.println("Feed forward for layer: " + i + " with value " + Arrays
				.deepToString(input.toDoubleArray()));
		}

		return input;
	}

	/**
	 * Provides an implementation of SGD for this neural network.
	 *
	 * @param training  a Collections object with Matrix[] objects, Matrix[0] is the data, Matrix[1]
	 *                  is the label.
	 * @param test      a Collections object with Matrix[] objects, Matrix[0] is the data, Matrix[1]
	 *                  is the label.
	 * @param epochs    how many iterations are we doing SGD for
	 * @param batchSize how big is the batch size, typically 32.
	 */
	public void stochasticGradientDescent(@NotNull List<NetworkInput> training,
		@NotNull List<NetworkInput> test,
		int epochs,
		int batchSize) {

		int trDataSize = training.size();
		int teDataSize = test.size();

		for (int i = 0; i < epochs; i++) {

			Collections.shuffle(training);

			System.out.println("Calculating epoch: " + (i + 1) + ".");

			for (int j = 0; j < trDataSize - batchSize; j += batchSize) {
				calculateMiniBatch(training.subList(j, j + batchSize));
			}

			List<NetworkInput> feedForwardData = this.feedForwardData(test);
			int correct = this.evaluationFunction.evaluatePrediction(feedForwardData).intValue();
			double loss = errorFunction.calculateCostFunction(feedForwardData);

			xValues.add((double) i);
			lossValues.add(loss);
			correctValues.add((double) correct);

			System.out.println("Loss: " + loss);

			System.out.println("Epoch " + (i + 1) + ": " + correct + "/" + teDataSize);
		}

	}

	/**
	 * Prints the networks performance in terms of loss and correct identification to a path.
	 * Example usage: "Users/{name}/Programming/DeepLearning/NN/Output/".
	 *
	 * @param basePath base path to image root.
	 *
	 * @return successful
	 */
	public boolean outputChart(String basePath) {
		XYChart lossToEpoch = QuickChart
			.getChart("Loss plot", "Epoch", "Loss", "loss(x)", xValues, lossValues);
		XYChart correctToEpoch = QuickChart
			.getChart("Correct plot", "Epoch", "Correct", "corr(x)", xValues, correctValues);

		// TODO: Fix with suffix / and change to loss/corr instead.

		try {
			BitmapEncoder.saveBitmapWithDPI(lossToEpoch, basePath + ThreadLocalRandom.current()
				.nextLong() + "_.jpg", BitmapFormat.PNG, 300);
			BitmapEncoder.saveBitmapWithDPI(correctToEpoch, basePath + ThreadLocalRandom
				.current().nextLong() + "_.jpg", BitmapFormat.PNG, 300);
			return true;
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
	}

	private List<NetworkInput> feedForwardData(List<NetworkInput> test) {
		List<NetworkInput> copy = new ArrayList<>();

		for (NetworkInput networkInput : test) {

			DenseMatrix out = this.feedForward(networkInput.getData());
			NetworkInput newOut = new NetworkInput(out, networkInput.getLabel());
			copy.add(newOut);
		}

		return copy;
	}
	// END MUTATORS

	public double getScore() {
		return this.score;
	}

	public void writeObject(String path) {
		File file;
		path = path.endsWith("/") ? path.substring(0, path.length() - 1) : path;

		try {
			FileOutputStream fs = new FileOutputStream(
				file = new File(
					path + "/NeuralNetwork_" + ThreadLocalRandom.current().nextLong() + "_.ser"));
			ObjectOutputStream os = new ObjectOutputStream(fs);
			os.writeObject(this);

			os.close();
			fs.close();

			System.out.println("Completed serialisation, see file: " + file.getPath());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
