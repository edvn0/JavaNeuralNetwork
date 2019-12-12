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
import math.activations.LinearFunction;
import math.activations.SoftmaxFunction;
import math.errors.CostFunction;
import math.errors.CrossEntropyCostFunction;
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
 * A class which can be both a single layer perceptron, and at the same time: an artifical deep
 * fully connected neural network. This implementation uses matrices to solve the problem of
 * learning and predicting on data.
 */
public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 7008674899707436812L;

	// Learning rate
	private double learningRate;

	// All activation functions for all layers
	private final ActivationFunction[] functions;

	// The error function to minimize.
	private final CostFunction costFunction;

	// The function to evaluate the data set.
	private final EvaluationFunction evaluationFunction;

	private DenseMatrix[] weights;
	private DenseMatrix[] vDw;
	private DenseMatrix[] sDw;
	private DenseMatrix[] epsilon;
	private DenseMatrix[] biases;

	// Helper field to hold the total amount of layers
	private final int totalLayers;

	// The structure of the network
	private int[] sizes;

	private static transient final ArrayList<Double> xValues = new ArrayList<>();
	private static transient final ArrayList<Double> lossValues = new ArrayList<>();
	private static transient final ArrayList<Double> correctValues = new ArrayList<>();

	/**
	 * Create a Neural Network with a learning rate, all the activation functions for all layers,
	 * the error function and the function to evaluate the network, and also the sizes of the
	 * layers, for example:
	 *
	 * int[] sizes = {3,4,4,1} is a 4-layered fully connected network with 3 input nodes, 1 output
	 * nodes, 2 hidden layers with 4 nodes in each of them.
	 *
	 * @param learning      a double representing step size in back propagation.
	 * @param functions     the activation functions for all layers
	 * @param errorFunction the error function to calculate error of last layers
	 * @param eval          the evaluation function to compare the network to the data's labels
	 * @param sizes         the table to initialize layers and weights.
	 */
	public NeuralNetwork(final double learning, final ActivationFunction[] functions,
		final CostFunction errorFunction,
		final EvaluationFunction eval, final int[] sizes) {
		this.learningRate = learning;
		this.functions = functions;
		this.costFunction = errorFunction;
		this.totalLayers = sizes.length;
		this.evaluationFunction = eval;
		this.sizes = sizes;

		initialiseBiases(sizes);
		initialiseWeights(sizes);

		if (errorFunction instanceof CrossEntropyCostFunction
			&& !(functions[functions.length - 1] instanceof SoftmaxFunction)) {
			throw new BackpropagationError(
				"To properly function, back-propagation needs the activation function of the last "
					+ "layer to be differentiable with respect to the error function.");
		}
	}

	public NeuralNetwork(NetworkBuilder b) {
		this.sizes = b.structure;
		this.functions = b.getActivationFunctions();
		this.costFunction = b.costFunction;
		this.learningRate = b.learningRate;
		this.evaluationFunction = b.evaluationFunction;
		this.totalLayers = sizes.length;

		initialiseBiases(sizes);
		initialiseWeights(sizes);
	}

	private void initialiseAdam() {
		for (int i = 0; i < this.weights.length; i++) {
			this.vDw[i] = Matrix.Factory.zeros(this.weights[i].getRowCount(), 1);
			this.sDw[i] = Matrix.Factory.zeros(this.weights[i].getRowCount(), 1);
			this.epsilon[i] = (DenseMatrix) Matrix.Factory.zeros(this.weights[i].getRowCount(), 1)
				.plus(10e-8);
		}
	}

	private void initialiseWeights(final int[] sizes) {
		this.weights = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			final int size = sizes[i];
			this.weights[i] = MatrixUtilities.map(Matrix.Factory.rand(sizes[i + 1], sizes[i]),
				(e) -> this.xavierInitialization(size));
		}
	}

	private void initialiseBiases(final int[] sizes) {
		this.biases = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.biases[i] = (DenseMatrix) Matrix.Factory.zeros(sizes[i + 1], 1).plus(0.01);
		}
	}

	public double xavierInitialization(final int prev) {
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
	 * @return a deserialised object.
	 *
	 * @throws IOException if file could not be found.
	 */
	public static NeuralNetwork readObject(String path) throws IOException {
		NeuralNetwork network = null;
		File file;
		path = (path.endsWith(".ser") ? path : path + ".ser");

		try (FileInputStream fs = new FileInputStream(file = new File(path));
			ObjectInputStream os = new ObjectInputStream(fs)) {

			network = (NeuralNetwork) os.readObject();

			System.out.println("Completed deserialization from file: " + file.getPath());
		} catch (final ClassNotFoundException e) {
			e.printStackTrace();
		}
		if (null != network) {
			return network;
		} else {
			throw new IOException("Something bad happened during deserialization.");
		}
	}

	public static NeuralNetwork readObject(final File file) throws IOException {
		NeuralNetwork neuralNetwork = null;
		try (FileInputStream fs = new FileInputStream(
			file); ObjectInputStream stream = new ObjectInputStream(fs)) {
			neuralNetwork = (NeuralNetwork) stream.readObject();

			System.out.println("Completed deserialization, see file: " + file.getAbsolutePath());
		} catch (final ClassNotFoundException e) {
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
		final String out = path.endsWith("/") ? path.substring(0, path.length() - 1) : path;

		try {
			final FileOutputStream fs = new FileOutputStream(
				file = new File(out + "/NeuralNetwork_" + getNow() + "_.ser"));
			final ObjectOutputStream os = new ObjectOutputStream(fs);
			os.writeObject(this);

			os.close();
			fs.close();

			System.out.println("Completed serialisation, see file: " + file.getPath());
		} catch (final IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Train the network with one example.
	 *
	 * @param input a {@link NetworkInput} object to be trained on.
	 */
	public void train(final NetworkInput input) {
		calculateMiniBatch(Collections.singletonList(input));
	}

	private void calculateMiniBatch(final List<NetworkInput> subList) {
		final int size = subList.size();

		final double scaleFactor = this.learningRate / size;

		final DenseMatrix[] dB = new DenseMatrix[this.totalLayers - 1];
		final DenseMatrix[] dW = new DenseMatrix[this.totalLayers - 1];
		for (int i = 0; i < this.totalLayers - 1; i++) {
			final DenseMatrix bias = getBias(i);
			final DenseMatrix weight = getWeight(i);
			dB[i] = Matrix.Factory.zeros(bias.getRowCount(), bias.getColumnCount());
			dW[i] = Matrix.Factory.zeros(weight.getRowCount(), weight.getColumnCount());
		}

		for (final NetworkInput data : subList) {
			final DenseMatrix dataIn = data.getData();
			final DenseMatrix label = data.getLabel();
			final List<DenseMatrix[]> deltas = backPropagate(dataIn, label);
			final DenseMatrix[] deltaB = deltas.get(0);
			final DenseMatrix[] deltaW = deltas.get(1);

			for (int j = 0; j < this.totalLayers - 1; j++) {
				dB[j] = (DenseMatrix) dB[j].plus(deltaB[j]);
				dW[j] = (DenseMatrix) dW[j].plus(deltaW[j]);
			}
		}

		for (int i = 0; i < dB.length; i++) {
			dB[i] = (DenseMatrix) dB[i].times(scaleFactor);
			dW[i] = (DenseMatrix) dW[i].times(scaleFactor);
		}

		for (int i = 0; i < this.totalLayers - 1; i++) {
			this.weights[i] = (DenseMatrix) this.weights[i].minus(dW[i]);
			this.biases[i] = (DenseMatrix) this.biases[i].minus(dB[i]);
		}
	}

	private List<DenseMatrix[]> backPropagate(final DenseMatrix toPredict,
		final DenseMatrix correct) {

		final List<DenseMatrix[]> totalDeltas = new ArrayList<>();

		final DenseMatrix[] deltaBiases = this.initializeDeltas(biases);
		final DenseMatrix[] deltaWeights = this.initializeDeltas(weights);

		// Perform Feed Forward here...
		final List<DenseMatrix> activations = new ArrayList<>();
		final List<DenseMatrix> xVector = new ArrayList<>();

		// Alters all arrays and lists.
		this.backPropFeedForward(toPredict, activations, xVector);
		// End feedforward

		// Calculate error signal for last layer

		// Applies the error function to the last layer, create
		DenseMatrix a = activations.get(activations.size() - 1);

		DenseMatrix deltaError = costFunction
			.applyErrorFunctionGradient(a, correct);

		// Iterate over all layers, they are indexed by the last layer (here given b
		for (int k = deltaBiases.length - 1; k >= 0; k--) {
			final DenseMatrix aCurr = activations.get(k + 1); // this layer
			final DenseMatrix aNext = activations.get(k); // Previous layer
			DenseMatrix differentiate = this.functions[k + 1].derivativeOnInput(aCurr, deltaError);

			deltaBiases[k] = differentiate;
			deltaWeights[k] = (DenseMatrix) differentiate
				.mtimes(aNext.transpose());

			deltaError = (DenseMatrix) this.weights[k].transpose().mtimes(differentiate);
		}

		totalDeltas.add(deltaBiases);
		totalDeltas.add(deltaWeights);

		return totalDeltas;
	}

	private DenseMatrix[] initializeDeltas(final DenseMatrix[] toCopyFrom) {
		final DenseMatrix[] deltas = new DenseMatrix[toCopyFrom.length];
		for (int i = 0; i < deltas.length; i++) {
			final int rows = (int) toCopyFrom[i].getRowCount();
			final int cols = (int) toCopyFrom[i].getColumnCount();
			deltas[i] = Matrix.Factory.zeros(rows, cols);
		}
		return deltas;
	}

	private void backPropFeedForward(final DenseMatrix starter, final List<DenseMatrix> actives,
		final List<DenseMatrix> vectors) {

		DenseMatrix toPredict = starter;

		actives.add(toPredict);
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			final DenseMatrix x = (DenseMatrix) this.weights[i]
				.mtimes(toPredict)
				.plus(this.biases[i]);

			vectors.add(x);

			toPredict = this.functions[i + 1].applyFunction(x);
			actives.add(toPredict);
		}
	}

	// -------------------------
	// Mutators
	// -------------------------
	private DenseMatrix[] getWeights() {
		return this.weights;
	}

	private DenseMatrix[] getBiasesAsMatrices() {
		final DenseMatrix[] biases = new DenseMatrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			biases[i] = getBias(i);
		}
		return biases;
	}

	private void setWeight(final int i, final DenseMatrix newWeights) {
		this.weights[i] = newWeights;
	}

	private DenseMatrix getWeight(final int i) {
		return this.weights[i];
	}

	private DenseMatrix getBias(final int i) {
		return this.biases[i];
	}

	private void setBias(final int i, final DenseMatrix outputMatrix) {
		this.biases[i] = outputMatrix;
	}

	/**
	 * Predict a single example input data.
	 *
	 * @param in {@link DenseMatrix} a Matrix to feed forward.
	 *
	 * @return a classification of {@link DenseMatrix}
	 */
	public DenseMatrix predict(final DenseMatrix in) {
		return feedForward(in);
	}

	/**
	 * Feed the input through the network for classification.
	 *
	 * @param in values to predict
	 *
	 * @return classified values.
	 */
	private DenseMatrix feedForward(final DenseMatrix in) {
		// Make input into matrix.
		DenseMatrix input = in;
		final DenseMatrix[] weights = getWeights();
		final DenseMatrix[] biases = getBiasesAsMatrices();
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
	 *                  NetworkInput.getData() is the data, NetworkInput.getLabel() is the label.
	 * @param test      a Collections object with {@link NetworkInput} objects,
	 *                  NetworkInput.getData() is the data, NetworkInput.getLabel() is the label.
	 * @param epochs    how many iterations are we doing SGD for
	 * @param batchSize how big is the batch size, typically 32. See https://stats.stackexchange.com/q/326663
	 */
	public void stochasticGradientDescent(@NotNull final List<NetworkInput> training,
		@NotNull final List<NetworkInput> test, final int epochs, final int batchSize) {

		// How many times will we decrease the learning rate?
		final int decreaseLR = epochs / 5;
		final int teDataSize = test.size();
		final int trDataSize = training.size();

		// Feed forward the test data
		final List<NetworkInput> ffD = this.feedForwardData(test);
		// Evaluate prediction with the interface EvaluationFunction.
		final int c = this.evaluationFunction.evaluatePrediction(ffD)
			.intValue();
		final double l = costFunction.calculateCostFunction(ffD);
		addPlotData(0, c, l);
		System.out.println("Loss: " + l);
		System.out.println("Epoch " + (0) + ": " + c + "/" + teDataSize);

		for (int i = 0; i < epochs; i++) {
			// Randomize training sample.
			Collections.shuffle(training);

			System.out.println("Calculating epoch: " + (i + 1) + ".");

			// Do backpropagation.
			for (int j = 0; j < trDataSize - batchSize; j += batchSize) {
				calculateMiniBatch(training.subList(j, j + batchSize));
			}

			// Feed forward the test data
			final List<NetworkInput> feedForwardData = this.feedForwardData(test);

			// Evaluate prediction with the interface EvaluationFunction.
			final int correct = this.evaluationFunction.evaluatePrediction(feedForwardData)
				.intValue();
			// Calculate loss with the interface CostFunction
			final double loss = costFunction.calculateCostFunction(feedForwardData);

			// Add the plotting data, x, y_1, y_2 to the global
			// lists of xValues, correctValues, lossValues.
			addPlotData((i + 1), correct, loss);

			// Lower learning rate for each k-th iteration.
			if ((i + 1) % decreaseLR == 0) {
				this.learningRate = this.learningRate * 0.95;
			}

			System.out.println("Loss: " + loss);
			System.out.println("Epoch " + (i + 1) + ": " + correct + "/" + teDataSize);

			// Lower learning rate each iteration?. Might implement? Don't know how to.
			// ADAM? Is that here? Are they different algorithms all together?
			// TODO: Implement Adam, RMSProp, Momentum?
			// this.learningRate = i % 10 == 0 ? this.learningRate / 4 : this.learningRate;
		}

	}

	// TODO: Implement ADAM.
	public void adam(final List<NetworkInput> training, final List<NetworkInput> testing,
		final int epochs,
		final int batchSize) {
		final int trainSize = training.size();
		final int testSize = training.size();

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

		final XYChart lossToEpoch = generateChart("Loss/Epoch", "Epoch", "Loss", "loss(x)", xValues,
			lossValues);

		final XYChart correctToEpoch = generateChart("Correct/Epoch", "Epoch", "Correct",
			"correct(x)", xValues,
			correctValues);

		final String use = basePath.endsWith("/") ? basePath : basePath + "/";
		final String loss = use + "LossToEpochPlot";
		final String correct = use + "CorrectToEpochPlot";

		final String now = getNow();

		final String nowLoss = loss + "_" + now;
		final String nowCorr = correct + "_" + now;

		try {
			BitmapEncoder.saveBitmapWithDPI(lossToEpoch, nowLoss, BitmapFormat.PNG, 300);
			BitmapEncoder.saveBitmapWithDPI(correctToEpoch, nowCorr, BitmapFormat.PNG, 300);
		} catch (final IOException e) {
			e.printStackTrace();
		}
	}

	public static String getNow() {
		String formattedDate;
		final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss", Locale.ENGLISH);
		formattedDate = sdf.format(new Date());
		return formattedDate;
	}

	private XYChart generateChart(final String heading, final String xLabel, final String yLabel,
		final String function,
		final List<Double> xValues, final List<Double> yValues) {
		final XYChart chart = QuickChart
			.getChart(heading, xLabel, yLabel, function, NeuralNetwork.xValues, yValues);
		chart.getStyler().setXAxisMin(0d);
		chart.getStyler().setXAxisMax(Collections.max(xValues));
		chart.getStyler().setYAxisMin(0d);
		chart.getStyler().setYAxisMax(Collections.max(yValues));
		return chart;
	}

	private List<NetworkInput> feedForwardData(final List<NetworkInput> test) {
		final List<NetworkInput> copy = new ArrayList<>();

		for (final NetworkInput networkInput : test) {

			final DenseMatrix out = this.feedForward(networkInput.getData());
			final NetworkInput newOut = new NetworkInput(out, networkInput.getLabel());
			copy.add(newOut);
		}

		return copy;
	}

	public double evaluateTestData(final List<NetworkInput> imagesTest, int size) {
		double sum = 0;
		for (int i = 0; i < size; i++) {
			final List<NetworkInput> test = this.feedForwardData(imagesTest);
			sum += evaluationFunction.evaluatePrediction(test).intValue();
		}

		return sum / size;

	}

	public static class NetworkBuilder {

		private final int[] structure;
		private int index;
		List<ActivationFunction> functions;
		private double learningRate;
		private CostFunction costFunction;
		private EvaluationFunction evaluationFunction;

		public NetworkBuilder(int[] structure) {
			this.structure = structure;
			this.index = 0;
			functions = new ArrayList<>();
		}

		public NetworkBuilder(int s) {
			this.structure = new int[s];
			this.index = 0;
			functions = new ArrayList<>();
		}

		public NetworkBuilder setFirstLayer(final int i) {
			structure[index] = i;
			functions.add(new LinearFunction());
			this.index++;
			return this;
		}

		public NetworkBuilder setLayer(final int i, final ActivationFunction f) {
			structure[index] = i;
			functions.add(f);
			this.index++;
			return this;
		}

		public NetworkBuilder setActivationFunction(ActivationFunction f) {
			this.functions.add(f);
			this.index++;
			return this;
		}

		public NetworkBuilder setLearningRate(double l) {
			this.learningRate = l;
			return this;
		}

		public NetworkBuilder setCostFunction(CostFunction k) {
			this.costFunction = k;
			return this;
		}

		public NetworkBuilder setEvaluationFunction(EvaluationFunction f) {
			this.evaluationFunction = f;
			return this;
		}

		public ActivationFunction[] getActivationFunctions() {
			ActivationFunction[] f = new ActivationFunction[this.index];

			if (this.functions.size() == this.structure.length) {
				// We have one too many functions, one associated with the "first layer"
				// which in essence does not exist, we apply a linear function here.
				// However, this is never calculated.
				for (int i = 1; i < this.structure.length; i++) {
					f[i] = this.functions.get(i);
				}
				// We do not care about this one, never gets evaluated.
				f[0] = new LinearFunction();
			} else if (this.functions.size() + 1 == this.structure.length) {
				// We have supplied the builder with the correct amount of functions.
				for (int i = 0; i < this.structure.length; i++) {
					f[i] = this.functions.get(i);
				}
			} else {
				throw new IllegalArgumentException("Not enough activation functions provided.");
			}
			return f;
		}

		public NetworkBuilder setLastLayer(final int i, final ActivationFunction f) {
			this.structure[index] = i;
			functions.add(f);
			index++;
			return this;
		}
	}
}
