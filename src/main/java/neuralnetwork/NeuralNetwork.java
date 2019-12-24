package neuralnetwork;

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
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import math.activations.ActivationFunction;
import math.activations.LinearFunction;
import math.activations.SoftmaxFunction;
import math.errors.CostFunction;
import math.errors.CrossEntropyCostFunction;
import math.evaluation.EvaluationFunction;
import me.tongfei.progressbar.ProgressBar;
import optimizers.Optimizer;
import optimizers.StochasticGradientDescent;
import org.jetbrains.annotations.NotNull;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.interfaces.Clearable;
import utilities.MatrixUtilities;
import utilities.NetworkUtilities;

/**
 * A class which can be both a single layer perceptron, and at the same time: an artifical deep
 * fully connected neural network. This implementation uses matrices to solve the problem of
 * learning and predicting on data.
 */
public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 7008674899707436812L;

	// All activation functions for all layers
	private final ActivationFunction[] functions;
	// The error function to minimize.
	private final CostFunction costFunction;
	// The function to evaluate the data set.
	private final EvaluationFunction evaluationFunction;
	// The optimizer to be used
	private final Optimizer optimizer;
	// Weights and biases of the network
	private DenseMatrix[] weights;
	private DenseMatrix[] biases;
	// Deltas and gradients for back-propagation.
	private DenseMatrix[] deltaWeights;
	private DenseMatrix[] deltaBiases;
	private DenseMatrix[] dW;
	private DenseMatrix[] dB;
	// Helper field to hold the total amount of layers
	private final int totalLayers;
	// The structure of the network
	private int[] sizes;

	// Members which supply functionality to the plots.
	private static transient final ArrayList<Double> xValues = new ArrayList<>();
	private static transient final ArrayList<Double> lossValues = new ArrayList<>();
	private static transient final ArrayList<Double> correctValues = new ArrayList<>();
	private static transient final ArrayList<Double> calculationTimes = new ArrayList<>();


	/**
	 * Create a Neural Network with a learning rate, all the activation functions for all layers,
	 * the error function and the function to evaluate the network, and also the sizes of the
	 * layers, for example:
	 *
	 * int[] sizes = {3,4,4,1} is a 4-layered fully connected network with 3 input nodes, 1 output
	 * nodes, 2 hidden layers with 4 nodes in each of them.
	 *
	 * @param functions     the activation functions for all layers
	 * @param errorFunction the error function to calculate error of last layers
	 * @param eval          the evaluation function to compare the network to the data's labels
	 * @param sizes         the table to initialize layers and weights.
	 */
	public NeuralNetwork(final double learning, final ActivationFunction[] functions,
		final CostFunction errorFunction,
		final EvaluationFunction eval, final int[] sizes) {
		this.functions = functions;
		this.costFunction = errorFunction;
		this.totalLayers = sizes.length;
		this.evaluationFunction = eval;
		this.sizes = sizes;
		this.optimizer = new StochasticGradientDescent(learning);

		initialiseBiases(sizes);
		this.deltaWeights = initializeMatrices(this.weights);
		initialiseWeights(sizes);
		this.deltaBiases = initializeMatrices(this.biases);

		if (errorFunction instanceof CrossEntropyCostFunction
			&& !(functions[functions.length - 1] instanceof SoftmaxFunction)) {
			throw new IllegalArgumentException(
				"To properly function, back-propagation needs the activation function of the last "
					+ "layer to be differentiable with respect to the error function.");
		}
	}

	public NeuralNetwork(NetworkBuilder b) {
		this.sizes = b.structure;
		this.functions = b.getActivationFunctions();
		this.costFunction = b.costFunction;
		this.evaluationFunction = b.evaluationFunction;
		this.totalLayers = sizes.length;

		// Initialize the optimizer and the parameters.
		this.optimizer = b.optimizer;
		this.optimizer.initializeOptimizer(this.totalLayers);

		// Initialize weights, deltas and gradients.
		initialiseWeights(sizes);
		this.deltaWeights = initializeMatrices(this.weights);
		this.dW = initializeMatrices(this.weights);
		initialiseBiases(sizes);
		this.deltaBiases = initializeMatrices(this.biases);
		this.dB = initializeMatrices(this.biases);
	}

	/**
	 * Returns matrices of zeroes, of similar dimensions as provided in input.
	 *
	 * @param toCopyFrom input matrices of RowCount_iXColumnCount_i
	 *
	 * @return zero matrices with RowCount_iXColumnCount_i dimensions.
	 */
	private DenseMatrix[] initializeMatrices(final DenseMatrix[] toCopyFrom) {
		final DenseMatrix[] deltas = new DenseMatrix[toCopyFrom.length];
		for (int i = 0; i < deltas.length; i++) {
			final int rows = (int) toCopyFrom[i].getRowCount();
			final int cols = (int) toCopyFrom[i].getColumnCount();
			deltas[i] = Matrix.Factory.zeros(rows, cols);
		}
		return deltas;
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
	 * Train the network with one example.
	 *
	 * @param input a {@link NetworkInput} object to be trained on.
	 */
	public void train(final NetworkInput input) {
		evaluateTrainingExample(Collections.singletonList(input));
		learnFromDeltas();
	}

	/**
	 * Back-propagates a data set and normalizes the deltas against the size of the batch to be used
	 * in an optimizer.
	 */
	private void evaluateTrainingExample(final List<NetworkInput> subList) {
		final int size = subList.size();

		for (final NetworkInput data : subList) {
			final List<DenseMatrix[]> deltas = backPropagate(data);
			final DenseMatrix[] deltaB = deltas.get(0);
			final DenseMatrix[] deltaW = deltas.get(1);

			for (int j = 0; j < this.totalLayers - 1; j++) {
				this.dW[j] = (DenseMatrix) this.dW[j].plus(deltaW[j].times(1d / size));
				this.dB[j] = (DenseMatrix) this.dB[j].plus(deltaB[j].times(1d / size));
			}
		}
	}

	/**
	 * Evaluates one example for multi threaded gradient descent.
	 */
	private void evaluateTrainingExample(final NetworkInput ni) {
		final List<DenseMatrix[]> deltas = backPropagate(ni);
		final DenseMatrix[] deltaB = deltas.get(0);
		final DenseMatrix[] deltaW = deltas.get(1);

		for (int j = 0; j < this.totalLayers - 1; j++) {
			dW[j] = (DenseMatrix) dW[j].plus(deltaW[j]);
			dB[j] = (DenseMatrix) dB[j].plus(deltaB[j]);
		}
	}

	/**
	 * Updates weights and biases and resets the batch adjusted deltas.
	 */
	private void learnFromDeltas() {
		this.weights = this.optimizer.changeWeights(this.weights, this.dW);
		this.biases = this.optimizer.changeBiases(this.biases, this.dB);
		Stream.of(dB).forEach(Clearable::clear);
		Stream.of(dW).forEach(Clearable::clear);
	}

	private List<DenseMatrix[]> backPropagate(NetworkInput in) {

		final List<DenseMatrix[]> totalDeltas = new ArrayList<>();
		final List<DenseMatrix> activations = new ArrayList<>();

		Stream.of(this.deltaBiases).forEach(Clearable::clear);
		Stream.of(this.deltaWeights).forEach(Clearable::clear);

		// Alters all arrays and lists.
		this.feedForward(in.getData(), activations);
		// End feedforward

		DenseMatrix a = activations.get(activations.size() - 1);
		DenseMatrix deltaError = costFunction
			.applyCostFunctionGradient(a, in.getLabel());

		// Iterate over all layers, they are indexed by the last layer (here given b
		for (int k = deltaBiases.length - 1; k >= 0; k--) {
			final DenseMatrix aCurr = activations.get(k + 1); // this layer
			final DenseMatrix aNext = activations.get(k); // Previous layer
			DenseMatrix differentiate = this.functions[k + 1]
				.derivativeOnInput(aCurr, deltaError);

			this.deltaBiases[k] = differentiate;
			this.deltaWeights[k] = (DenseMatrix) differentiate
				.mtimes(aNext.transpose());

			deltaError = (DenseMatrix) this.weights[k].transpose().mtimes(differentiate);
		}

		totalDeltas.add(deltaBiases);
		totalDeltas.add(deltaWeights);

		return totalDeltas;
	}

	private void feedForward(final DenseMatrix starter,
		final List<DenseMatrix> actives) {
		DenseMatrix toPredict = starter;
		actives.add(toPredict);
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			final DenseMatrix x =
				(DenseMatrix) this.weights[i]
					.mtimes(toPredict)
					.plus(this.biases[i]);

			toPredict = this.functions[i + 1].applyFunction(x);
			actives.add(toPredict);
		}
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
		DenseMatrix input = in;
		for (int i = 0; i < this.totalLayers - 1; i++) {
			input = functions[i + 1]
				.applyFunction((DenseMatrix) this.weights[i].mtimes(input).plus(this.biases[i]));
		}
		return input;
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
		double s = imagesTest.size();
		for (int i = 0; i < size; i++) {
			final List<NetworkInput> test = this.feedForwardData(imagesTest);
			sum += evaluationFunction.evaluatePrediction(test).intValue() / s;
		}

		return sum;
	}

	/**
	 * Trains this network on training data, and validates on validation data. Uses a {@link
	 * Optimizer} to optimize the gradient descent.
	 *
	 * @param training   a Collections object with {@link NetworkInput} objects,
	 *                   NetworkInput.getData() is the data, NetworkInput.getLabel() is the label.
	 * @param validation a Collections object with {@link NetworkInput} objects,
	 *                   NetworkInput.getData() is the data, NetworkInput.getLabel() is the label.
	 * @param epochs     how many iterations are we doing the descent for
	 * @param batchSize  how big is the batch size, typically 32. See https://stats.stackexchange.com/q/326663
	 */
	public void train(@NotNull final List<NetworkInput> training,
		@NotNull final List<NetworkInput> validation, final int epochs, final int batchSize) {

		// How many times will we decrease the learning rate?
		List<List<NetworkInput>> split = NetworkUtilities.splitData(training, batchSize);

		// Feed forward the validation data prior to the batch descent
		// to establish
		final List<NetworkInput> ffD = this.feedForwardData(validation);
		// Evaluate prediction with the interface EvaluationFunction.
		int correct = this.evaluationFunction.evaluatePrediction(ffD)
			.intValue();
		double loss = this.costFunction.calculateCostFunction(ffD);
		addPlotData(0, correct, loss);

		final ProgressBar bar = new ProgressBar("Backpropagation", epochs);
		for (int i = 0; i < epochs; i++) {
			// Randomize training sample.
			// TODO: is this necessary????
			Collections.shuffle(split);
			split.forEach(Collections::shuffle);

			// Calculates a batch of training data and update the deltas.
			long t1, t2;

			t1 = System.nanoTime();
			for (int k = 0; k <= training.size() / batchSize; k++) {
				getBatch(k, training, batchSize).parallelStream()
					.forEach(this::evaluateTrainingExample);
				learnFromDeltas();
			}
			t2 = System.nanoTime();

			/* TODO: Keeping this here to show to myself what is
			     "correct"; this parallelStreaming thing, I do not trust it.
			for (List<NetworkInput> in : split) {
				evaluateTrainingExample(in);
				learnFromDeltas();
			}*/

			// Feed forward the test data
			final List<NetworkInput> feedForwardData = this.feedForwardData(validation);

			// Evaluate prediction with the interface EvaluationFunction.
			correct = this.evaluationFunction.evaluatePrediction(feedForwardData)
				.intValue();
			// Calculate loss with the interface CostFunction
			loss = this.costFunction.calculateCostFunction(feedForwardData);

			// Add the plotting data, x, y_1, y_2 to the global
			// lists of xValues, correctValues, lossValues.
			addPlotData((i + 1), correct, loss, (t2 - t1));
			bar.step();
		}
		bar.close();
	}


	private List<NetworkInput> getBatch(final int k, final List<NetworkInput> training,
		int batchSize) {
		int fromIx = k * batchSize;
		int toIx = Math.min(training.size(), (k + 1) * batchSize);
		return Collections.unmodifiableList(training.subList(fromIx, toIx));
	}

	// -------------------------------------------------------------
	// HERE STARTS AUXILIARIES, I/O for Plots, and for serialisation.
	// -------------------------------------------------------------

	/**
	 * Adds plotting data to global lists.
	 *
	 * @param i       the epoch
	 * @param correct the validation correctness for this epoch
	 * @param loss    the loss function for this epoch
	 * @param time    the time to calculate one epoch
	 */
	private void addPlotData(final double i, final double correct, final double loss,
		final long time) {
		xValues.add(i);
		lossValues.add(loss);
		correctValues.add(correct);
		calculationTimes.add((double) TimeUnit.NANOSECONDS.toMillis(time));
	}

	private void addPlotData(final int i, final int correct, final double loss) {
		xValues.add((double) i);
		lossValues.add(loss);
		correctValues.add((double) correct);
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

		List<Double> copy = xValues.subList(1, xValues.size());
		final XYChart calcTimeToEpoch = generateChart("Benchmark time/Epoch", "Epoch", "Time",
			"time(x)", copy,
			calculationTimes);

		final String use = basePath.endsWith("/") ? basePath : basePath + "/";
		final String loss = use + "LossToEpochPlot";
		final String correct = use + "CorrectToEpochPlot";
		final String calc = use + "BenchmarkTimingPlot";

		System.out.println(use);

		String formattedDate;
		final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss", Locale.ENGLISH);
		formattedDate = sdf.format(new Date());

		final String now = formattedDate;

		final String nowLoss = loss + "_" + now;
		final String nowCorr = correct + "_" + now;
		final String nowCalc = calc + "_" + now;

		try {
			BitmapEncoder.saveBitmapWithDPI(lossToEpoch, nowLoss, BitmapFormat.PNG, 300);
			BitmapEncoder.saveBitmapWithDPI(correctToEpoch, nowCorr, BitmapFormat.PNG, 300);
			BitmapEncoder.saveBitmapWithDPI(calcTimeToEpoch, nowCalc, BitmapFormat.PNG, 300);
		} catch (final IOException e) {
			e.printStackTrace();
		}
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

	/**
	 * Reads a .ser file or a path to a .ser file (with the extension excluded) to a NeuralNetwork
	 * object.
	 *
	 * E.g. /Users/{other paths}/NeuralNetwork_{LONG}_.ser works as well as /Users/{other
	 * paths}/NeuralNetwork_{LONG}_
	 *
	 * @param file the file to read-
	 *
	 * @return a deserialised network.
	 *
	 * @throws IOException if file is not readable.
	 */
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

	/**
	 * Serialises this network. Outputs a file (.ser) with the date.
	 *
	 * @param path the path to the serialised file.
	 */
	public void writeObject(final String path) {
		File file;
		final String out = path.endsWith("/") ? path.substring(0, path.length() - 1) : path;
		String formattedDate;
		final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss", Locale.ENGLISH);
		formattedDate = sdf.format(new Date());

		try {
			final FileOutputStream fs = new FileOutputStream(
				file = new File(out + "/NeuralNetwork_" + formattedDate + "_.ser"));
			final ObjectOutputStream os = new ObjectOutputStream(fs);
			os.writeObject(this);

			os.close();
			fs.close();

			System.out.println("Completed serialisation, see file: " + file.getPath());
		} catch (final IOException e) {
			e.printStackTrace();
		}
	}

	private XYChart generateChart(final String heading, final String xLabel, final String yLabel,
		final String function,
		final List<Double> xValues, final List<Double> yValues) {
		final XYChart chart = QuickChart
			.getChart(heading, xLabel, yLabel, function, xValues, yValues);
		chart.getStyler().setXAxisMin(0d);
		chart.getStyler().setXAxisMax(Collections.max(xValues));
		chart.getStyler().setYAxisMin(0d);
		chart.getStyler().setYAxisMax(Collections.max(yValues));
		return chart;
	}

	/**
	 * A Builder for the Network.
	 */
	public static class NetworkBuilder {

		private int[] structure;
		private int index;
		private List<ActivationFunction> functions;
		private CostFunction costFunction;
		private EvaluationFunction evaluationFunction;
		private Optimizer optimizer;

		public NetworkBuilder(int[] structure) {
			this.structure = structure;
			this.index = 0;
			this.functions = new ArrayList<>();
		}

		public NetworkBuilder(int s) {
			this.structure = new int[s];
			this.index = 0;
			this.functions = new ArrayList<>();
		}

		public NetworkBuilder setFirstLayer(final int i) {

			structure[index] = i;
			this.index++;

			functions.add(new LinearFunction());
			return this;
		}

		public NetworkBuilder setOptimizer(Optimizer o) {
			this.optimizer = o;
			return this;
		}

		public NetworkBuilder setLayer(final int i, final ActivationFunction f) {

			structure[index] = i;
			this.index++;
			functions.add(f);
			return this;
		}

		public NetworkBuilder setActivationFunction(ActivationFunction f) {
			this.functions.add(f);
			this.index++;
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
			ActivationFunction[] f;
			f = new ActivationFunction[this.index];

			if (this.functions.size() == this.structure.length) {
				// We have one too many functions, one associated with the "first layer"
				// which in essence does not exist, we MNISTApply a linear function here.
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
			this.index++;
			this.functions.add(f);
			return this;
		}
	}
}
