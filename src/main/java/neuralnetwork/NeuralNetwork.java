package neuralnetwork;

import errors.BackpropagationError;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
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
public class NeuralNetwork implements Serializable {

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
	private DenseMatrix[] vDw;
	private DenseMatrix[] sDw;
	private DenseMatrix[] epsilon;

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
	 * @param learning  learning rate
	 * @param functions activation functions
	 * @param function  error function
	 * @param eval      evaluation function
	 * @param sizes     nodes in each layer
	 * @param score     score for the network
	 */
	public NeuralNetwork(double learning, ActivationFunction[] functions, ErrorFunction function,
		EvaluationFunction eval,
		int[] sizes, double score) {
		this(learning, functions, function, eval, sizes, 0, 0, 0);
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
	 * @param learning    a double representing step size in back propagation.
	 * @param functions   the activation functions for all layers
	 * @param function    the error function to calculate error of last layers
	 * @param eval        the evaluation function to compare the network to the data's labels
	 * @param sizes       the table to initialize layers and weights.
	 * @param adamBetaOne adam beta hyperparameter
	 * @param adamBetaTwo adam beta hyperparameter
	 * @param adamEpsilon adam beta hyperparameter
	 */
	public NeuralNetwork(double learning, ActivationFunction[] functions, ErrorFunction function,
		EvaluationFunction eval,
		int[] sizes, final double adamBetaOne, final double adamBetaTwo, final double adamEpsilon) {
		this.learningRate = learning;
		this.functions = functions;
		this.errorFunction = function;
		this.totalLayers = sizes.length;
		this.evaluationFunction = eval;
		this.score = 0;

		createLayers(sizes);
		initialiseWeights(sizes);
		initialiseAdam();

		if (function instanceof CrossEntropyErrorFunction &&
			!(functions[functions.length - 1] instanceof SoftmaxFunction)) {
			throw new BackpropagationError(
				"To properly function, back-propagation needs the activation function of the last "
					+ "layer to be differentiable with respect to the error function.");
		}
	}

	private void initialiseAdam() {
		for (int i = 0; i < this.weights.length; i++) {
			this.vDw[i] = Matrix.Factory.zeros(this.weights[i].getRowCount(), 1);
			this.sDw[i] = Matrix.Factory.zeros(this.weights[i].getRowCount(), 1);
			this.epsilon[i] = Matrix.Factory
				.importFromArray(generateStability(this.weights[i].getRowCount()));
		}

	}

	private double[][] generateStability(final long rowCount) {
		double[][] values = new double[(int) rowCount][1];
		for (int i = 0; i < rowCount; i++) {
			values[i][0] = 10e-8;
		}
		return values;
	}

	private void initialiseWeights(int[] sizes) {
		this.weights = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			final int size = sizes[i];
			this.weights[i] = MatrixUtilities
				.map(Matrix.Factory.rand(sizes[i + 1], sizes[i]),
					(e) -> this.xavierInitialization(size));
		}
	}

	private void createLayers(int[] sizes) {
		this.biases = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.biases[i] = Matrix.Factory.zeros(sizes[i + 1], 1);
		}
	}

	public double xavierInitialization(int prev) {
		return ThreadLocalRandom.current().nextGaussian() * (Math.sqrt(2) / Math.sqrt(prev));
	}

	private int getTotalLayers() {
		return this.totalLayers;
	}

	/**
	 * Reads a .ser file or a path to a .ser file (with the extension excluded) to a NeuralNetwork
	 * object.
	 *
	 * E.g. /Users/{other paths}/NeuralNetwork_{LONG}_.ser works as well as /Users/{other
	 * paths}/NeuralNetwork_{LONG}_
	 *
	 * @param path the full path to the file. does not require the .ser extension.
	 *
	 * @return a deserialized object.
	 *
	 * @throws IOException if file could not be found.
	 */
	public static NeuralNetwork readObject(String path) throws IOException {
		NeuralNetwork network = null;
		File file;
		path = (path.endsWith(".ser") ? path : path + ".ser");

		try (FileInputStream fs = new FileInputStream(
			file = new File(path)); ObjectInputStream os = new ObjectInputStream(fs)) {

			network = (NeuralNetwork) os.readObject();

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

	public static NeuralNetwork readObject(File file) throws IOException {
		NeuralNetwork neuralNetwork = null;
		try (FileInputStream fs = new FileInputStream(
			file); ObjectInputStream stream = new ObjectInputStream(fs)) {
			neuralNetwork = (NeuralNetwork) stream.readObject();

			System.out.println("Completed deserialization, see file: " + file.getAbsolutePath());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		if (null != neuralNetwork) {
			return neuralNetwork;
		} else {
			throw new IOException("Something bad happened during deserialization.");
		}
	}

	public void writeObject(final String path) {
		File file;
		String out = path.endsWith("/") ? path.substring(0, path.length() - 1) : path;

		try {
			FileOutputStream fs = new FileOutputStream(
				file = new File(
					out + "/NeuralNetwork_" + getNow() + "_.ser"));
			ObjectOutputStream os = new ObjectOutputStream(fs);
			os.writeObject(this);

			os.close();
			fs.close();

			System.out.println("Completed serialisation, see file: " + file.getPath());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Train the network with one example.
	 *
	 * @param input a {@link NetworkInput} object to be trained on.
	 */
	public void train(NetworkInput input) {
		calculateMiniBatch(Collections.singletonList(input));
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
			setBias(i, nB);
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

		// Alters all arrays and lists.
		this.backPropFeedForward(toPredict, activations, xVector);
		// End feedforward

		// Calculate error signal for last layer
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
		List<DenseMatrix> vectors) {

		DenseMatrix toPredict = starter;

		actives.add(toPredict);
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			DenseMatrix x = (DenseMatrix) this.weights[i].mtimes(toPredict).plus(this.biases[i]);
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

	private void setBias(int i, DenseMatrix outputMatrix) {
		this.biases[i] = outputMatrix;
	}

	/**
	 * Predict a single example input data.
	 *
	 * @param in {@link DenseMatrix} a Matrix to feed forward.
	 *
	 * @return a classification of {@link DenseMatrix}
	 */
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
	 * Provides an implementation of SGD for this neural network.
	 *
	 * @param training  a Collections object with {@link NetworkInput} objects,
	 *                  NetworkInput.getData() is the data, NetworkInput.getLabel()is the label.
	 * @param test      a Collections object with {@link NetworkInput} objects,
	 *                  NetworkInput.getData() is the data, NetworkInput.getLabel() is the label.
	 * @param epochs    how many iterations are we doing SGD for
	 * @param batchSize how big is the batch size, typically 32. See https://stats.stackexchange.com/q/326663
	 */
	public void stochasticGradientDescent(@NotNull List<NetworkInput> training,
		@NotNull List<NetworkInput> test,
		int epochs,
		int batchSize) {

		int trDataSize = training.size();
		int teDataSize = test.size();

		for (int i = 0; i < epochs; i++) {
			// Randomize training sample.
			Collections.shuffle(training);

			System.out.println("Calculating epoch: " + (i + 1) + ".");

			// Do backpropagation.
			for (int j = 0; j < trDataSize - batchSize; j += batchSize) {
				calculateMiniBatch(training.subList(j, j + batchSize));
			}

			// Feed forward the test data
			List<NetworkInput> feedForwardData = this.feedForwardData(test);

			// Evaluate prediction with the interface EvaluationFunction.
			int correct = this.evaluationFunction.evaluatePrediction(feedForwardData).intValue();
			// Calculate loss with the interface ErrorFunction
			double loss = errorFunction.calculateCostFunction(feedForwardData);

			// Add the plotting data, x, y_1, y_2 to the global
			// lists of xValues, correctValues, lossValues.
			addPlotData(i, correct, loss);

			System.out.println("Loss: " + loss);
			System.out.println("Epoch " + (i + 1) + ": " + correct + "/" + teDataSize);

			// Lower learning rate each iteration?. Might implement? Don't know how to.
			// ADAM? Is that here? Are they different algorithms all together?
			// TODO: Implement Adam, RMSProp, Momentum?
			// this.learningRate = i % 10 == 0 ? this.learningRate / 4 : this.learningRate;
		}

	}

	public void adam(List<NetworkInput> training, List<NetworkInput> testing, int epochs,
		int batchSize) {
		int trainSize = training.size();
		int testSize = training.size();

		for (int i = 0; i < epochs; i++) {

			Collections.shuffle(training);
			for (int k = 0; k < trainSize - batchSize; k += batchSize) {
				calculateMiniBatch(training.subList(k, k + batchSize));
			}

		}
	}

	private void addPlotData(final double i, final double correct, final double loss) {
		xValues.add(i);
		lossValues.add(loss);
		correctValues.add(correct);
	}

	/**
	 * Prints the networks performance in terms of loss and correct identification to a path.
	 * Example usage: "Users/{name}/Programming/DeepLearning/NN/Output/".
	 *
	 * @param basePath base path to image root.
	 */
	public void outputChart(final String basePath) {

		XYChart lossToEpoch = generateChart("Loss/Epoch", "Epoch", "Loss", "loss(x)",
			xValues,
			lossValues);

		XYChart correctToEpoch = generateChart("Correct/Epoch", "Epoch", "Correct", "correct(x)",
			xValues,
			correctValues);

		String use = basePath.endsWith("/") ? basePath : basePath + "/";
		String loss = use + "LossToEpochPlot";
		String correct = use + "CorrectToEpochPlot";

		String now = getNow();

		String nowLoss = loss + "_" + now;
		String nowCorr = correct + "_" + now;

		try {
			BitmapEncoder.saveBitmapWithDPI(lossToEpoch, nowLoss, BitmapFormat.PNG, 300);
			BitmapEncoder.saveBitmapWithDPI(correctToEpoch, nowCorr, BitmapFormat.PNG, 300);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static String getNow() {
		String formattedDate;
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss", Locale.ENGLISH);
		formattedDate = sdf.format(new Date());
		return formattedDate;
	}

	private XYChart generateChart(String heading, String xLabel, String yLabel, String function,
		List<Double> xValues,
		List<Double> yValues) {
		XYChart chart = QuickChart.getChart(heading, xLabel, yLabel, function,
			NeuralNetwork.xValues, yValues);
		chart.getStyler().setXAxisMin(0d);
		chart.getStyler().setXAxisMax(Collections.max(xValues));
		chart.getStyler().setYAxisMin(0d);
		chart.getStyler().setYAxisMax(Collections.max(yValues));
		return chart;
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


}
