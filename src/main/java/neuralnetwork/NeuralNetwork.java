package neuralnetwork;

import errors.BackpropagationError;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import math.activations.ActivationFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import matrix.Matrix;
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

	// The function to evaluate the data set.
	private EvaluationFunction evaluationFunction;

	// 0 based connections, i.e., connection 0 is from Layer 0 to Layer 1.
	private Matrix[] weights;

	// 0 based layering, i.e. index 0 in layers is layer 0.
	private Matrix[] biases;

	// Helper field to hold the total amount of layers
	private int totalLayers;

	// Current best score for this network, used for serialisation
	private double score;

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
		this.weights = new Matrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.weights[i] = Matrix.random(sizes[i + 1], sizes[i]);
		}
	}

	private void createLayers(int[] sizes) {
		this.biases = new Matrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			this.biases[i] = new Matrix(sizes[i + 1], 1);
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
	public void train(Matrix training, Matrix correct) {
		calculateMiniBatch(Collections.singletonList(new NetworkInput(training, correct)));
	}

	private void calculateMiniBatch(List<NetworkInput> subList) {
		int size = subList.size();
		Matrix[] dB = new Matrix[this.totalLayers - 1];
		Matrix[] dW = new Matrix[this.totalLayers - 1];
		for (int i = 0; i < this.totalLayers - 1; i++) {
			Matrix bias = getBias(i);
			Matrix weight = getWeight(i);
			dB[i] = new Matrix(bias.getRows(), bias.getColumns());
			dW[i] = new Matrix(weight.getRows(), weight.getColumns());
		}

		for (NetworkInput data : subList) {
			Matrix dataIn = data.getData();
			Matrix label = data.getLabel();
			List<Matrix[]> deltas = backPropagate(dataIn, label);
			Matrix[] deltaB = deltas.get(0);
			Matrix[] deltaW = deltas.get(1);

			for (int j = 0; j < this.totalLayers - 1; j++) {
				dB[j] = dB[j].add(deltaB[j]);
				dW[j] = dW[j].add(deltaW[j]);
			}
		}

		for (int i = 0; i < this.totalLayers - 1; i++) {
			Matrix cW = getWeight(i);
			Matrix cB = getBias(i);

			Matrix scaledDeltaB = dB[i].map((e) -> e * (this.learningRate / size));
			Matrix scaledDeltaW = dW[i].map((e) -> e * (this.learningRate / size));

			setWeight(i, cW.subtract(scaledDeltaW));
			setLayerBias(i, cB.subtract(scaledDeltaB));
		}
	}

	private List<Matrix[]> backPropagate(Matrix toPredict, Matrix correct) {
		List<Matrix[]> totalDeltas = new ArrayList<>();

		Matrix[] weights = getWeights();
		Matrix[] biases = getBiasesAsMatrices();

		Matrix[] deltaBiases = this.initializeDeltas(biases);
		Matrix[] deltaWeights = this.initializeDeltas(weights);

		// Perform Feed Forward here...
		List<Matrix> activations = new ArrayList<>();
		List<Matrix> xVector = new ArrayList<>();

		// Alters all arrays and lists.
		this.backPropFeedForward(toPredict, activations, xVector, weights, biases);
		// End feedforward

		// Calculate error signal for last layer
		Matrix error;
		Matrix deltaError;

		// Applies the error function to the last layer, create
		error = errorFunction
			.applyErrorFunctionGradient(correct, xVector.get(xVector.size() - 1));

		/*deltaError = error
			.hadamard(functions[activations.size() - 1]
				.applyDerivative(xVector.get(xVector.size() - 1), null));*/
		deltaError = error.hadamard(xVector.get(xVector.size() - 1));

		// Set the deltas to the error signals of bias and weight.
		deltaBiases[deltaBiases.length - 1] = deltaError;
		deltaWeights[deltaWeights.length - 1] = deltaError
			.multiply(activations.get(activations.size() - 2).transpose());

		// Now iteratively apply the rule
		for (int k = deltaBiases.length - 2; k >= 0; k--) {
			Matrix z = xVector.get(k);
			Matrix differentiate = functions[k + 1].applyDerivative(z, null);

			deltaError = weights[k + 1].transpose().multiply(deltaError)
				.hadamard(differentiate);

			deltaBiases[k] = deltaError;
			deltaWeights[k] = deltaError.multiply(activations.get(k).transpose());
		}
		totalDeltas.add(deltaBiases);
		totalDeltas.add(deltaWeights);

		return totalDeltas;
	}

	private Matrix[] initializeDeltas(Matrix[] toCopyFrom) {
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
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			Matrix x = weights[i].multiply(toPredict).add(biases[i]);
			vectors.add(x);

			toPredict = this.functions[i + 1].applyFunction(x, null);
			actives.add(toPredict);
		}
	}

	//-------------------------
	// Mutators
	//-------------------------
	private Matrix[] getWeights() {
		return this.weights;
	}

	private Matrix[] getBiasesAsMatrices() {
		Matrix[] biases = new Matrix[getTotalLayers() - 1];
		for (int i = 0; i < getTotalLayers() - 1; i++) {
			biases[i] = getBias(i);
		}
		return biases;
	}

	private void setWeight(int i, Matrix newWeights) {
		this.weights[i].setData(newWeights);
	}

	private Matrix getWeight(int i) {
		return this.weights[i];
	}

	private Matrix getBias(int i) {
		return this.biases[i];
	}

	private void setLayerBias(int i, Matrix outputMatrix) {
		this.biases[i] = outputMatrix;
	}

	@Override
	public Matrix predict(Matrix in) {
		return feedForward(in);
	}

	/**
	 * Feed the input through the network for classification.
	 *
	 * @param in values to predict
	 *
	 * @return classified values.
	 */
	private Matrix feedForward(Matrix in) {
		// Make input into matrix.
		Matrix input = in;
		Matrix[] weights = getWeights();
		Matrix[] biases = getBiasesAsMatrices();
		for (int i = 0; i < this.totalLayers - 1; i++) {
			input = functions[i].applyFunction(weights[i].multiply(input).add(biases[i]), null);
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
		@NotNull List<NetworkInput> test, int epochs,
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

			int correct = (int)
				this.evaluationFunction.evaluatePrediction(feedForwardData)
					.getElement(0, 0);

			System.out.println("Loss: " + errorFunction.calculateCostFunction(feedForwardData));

			this.score = (correct + 0.00001d) / teDataSize;
			System.out.println("Epoch " + (i + 1) + ": " + correct + "/" + teDataSize);
		}
	}

	private List<NetworkInput> feedForwardData(List<NetworkInput> test) {
		List<NetworkInput> copy = new ArrayList<>();
		for (NetworkInput networkInput : test) {
			Matrix out = this.feedForward(networkInput.getData());
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
		Date date = Calendar.getInstance().getTime();
		DateFormat dateFormat = new SimpleDateFormat("yyyy-mm-dd");
		String strDate = dateFormat.format(date).replace("-", "_");
		try {
			FileOutputStream fs = new FileOutputStream(
				file = new File(path + "/NeuralNetwork_" + strDate + ".ser"));
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
