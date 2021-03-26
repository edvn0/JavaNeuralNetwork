package neuralnetwork.layer;

import static java.util.stream.Collectors.toList;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.DeepLearnable;
import neuralnetwork.NetworkMetrics;
import neuralnetwork.initialiser.ParameterInitializer;
import neuralnetwork.inputs.NetworkInput;
import org.jetbrains.annotations.NotNull;
import utilities.types.Pair;

@Slf4j
public class LayeredNeuralNetwork<M> implements DeepLearnable<M> {

	public static final boolean DEBUG = false;

	private final List<NetworkLayer<M>> networkLayers;
	// The error function to minimize.
	private final CostFunction<M> costFunction;
	// The function to evaluate the data set.
	private final EvaluationFunction<M> evaluationFunction;
	// The optimizer to be used
	private final Optimizer<M> optimizer;

	private final ParameterInitializer<M> initializer;
	private final int inputNeurons;
	private boolean clipping;

	public LayeredNeuralNetwork(LayeredNetworkBuilder<M> b) {
		this.costFunction = b.costFunction;
		this.evaluationFunction = b.evaluationFunction;

		this.optimizer = b.optimizer;
		this.optimizer.initializeOptimizer(b.total, null, null);

		this.initializer = b.initializer;
		initializer.init(b.calculateStructure());

		this.clipping = b.gradientClipping;

		this.networkLayers = new ArrayList<>();

		var first = b.layers.get(0);
		this.inputNeurons = first.getNeurons();
		NetworkLayer<M> firstLayer = new NetworkLayer<>(first.getFunction(), first.getNeurons());
		this.networkLayers.add(firstLayer);

		var weights = initializer.getWeightParameters();
		var biases = initializer.getBiasParameters();
		var deltaW = initializer.getDeltaWeightParameters();
		var deltaB = initializer.getDeltaBiasParameters();

		var prev = firstLayer;
		for (int i = 1; i < b.total; i++) {
			var layer = new NetworkLayer<>(b.layers.get(i));

			var layerWeight = weights.get(i - 1);
			var layerBias = biases.get(i - 1);
			var layerDeltaWeight = deltaW.get(i - 1);
			var layerDeltaBias = deltaB.get(i - 1);

			layer.setWeight(layerWeight);
			layer.setBias(layerBias);
			layer.setDeltaWeight(layerDeltaWeight);
			layer.setDeltaBias(layerDeltaBias);
			layer.setRegularization(layer.getL2());

			layer.setPrecedingLayer(prev);
			this.networkLayers.add(layer);

			prev = layer;
		}
	}

	private LayeredNeuralNetwork(int inputNeurons, LayeredNetworkBuilder<M> b) {
		this.costFunction = b.costFunction;
		this.evaluationFunction = b.evaluationFunction;

		this.optimizer = b.optimizer;
		this.optimizer.initializeOptimizer(b.total, null, null);

		this.initializer = b.initializer;
		initializer.init(b.calculateStructure());

		this.networkLayers = new ArrayList<>();

		this.inputNeurons = inputNeurons;

		var first = b.layers.get(0);
		NetworkLayer<M> firstLayer = new NetworkLayer<>(first.getFunction(), first.getNeurons());
		this.networkLayers.add(firstLayer);

		for (int i = 1; i < b.total; i++) {
			var layer = new NetworkLayer<>(b.layers.get(i));
			this.networkLayers.add(layer);
		}
	}

	public static <U> LayeredNeuralNetwork<U> deserialize(int inputNeurons,
		LayeredNetworkBuilder<U> b) {
		return new LayeredNeuralNetwork<>(inputNeurons, b);
	}

	@Override
	public void train(List<NetworkInput<M>> training, final int epochs, int batchSize) {
		int batches = training.size() / batchSize;

		for (int i = 0; i < epochs; i++) {
			for (int j = 0; j <= batches; j++) {
				getBatch(j, batchSize, training).parallelStream()
					.forEach(e -> this.evaluate(e.getData(), e.getLabel()));
				this.fit();
			}
		}
	}

	private static <U> List<U> getBatch(final int i, final int batchSize, final List<U> data) {
		int fromIx = i * batchSize;
		int toIx = Math.min(data.size(), (i + 1) * batchSize);
		return Collections.unmodifiableList(data.subList(fromIx, toIx));
	}

	public void train(List<NetworkInput<M>> training, List<NetworkInput<M>> validation,
		final int epochs,
		final int batchSize) {

		int batches = training.size() / batchSize;

		log.info("Prior to training:");
		log.info("\nLoss: {}\nEvaluation percentage: {}%.", this.testLoss(validation),
			this.testEvaluation(validation, 37) * 100d);

		for (int epoch = 0; epoch < epochs; epoch++) {
			log.info("Epoch: {}", epoch + 1);
			for (int i = 0; i <= batches; i++) {
				getBatch(i, batchSize, training).parallelStream().forEach(e -> {
					this.evaluate(e.getData(), e.getLabel());
				});
				this.fit();
			}

			log.info("\nLoss: {}\nEvaluation percentage: {}%.", this.testLoss(validation),
				this.testEvaluation(validation, 37) * 100d);
		}
	}

	public void trainWithMetrics(@NotNull final List<NetworkInput<M>> training,
		@NotNull final List<NetworkInput<M>> validation, final int epochs, final int batchSize,
		final String path) {

		long t1, t2;
		// Members which supply functionality to the plots.
		final NetworkMetrics metrics = new NetworkMetrics(training.get(0).getData().name());
		// Feed forward the validation data prior to the batch descent
		// to establish a ground truth value
		final var ffD = this.feedforward(validation);
		// Evaluate prediction with the interface EvaluationFunction.
		double correct = this.evaluationFunction.evaluatePrediction(ffD);
		double loss = this.costFunction.calculateCostFunction(ffD);
		log.info("\n Ground truth before training: \n Loss: \t {}\n Correct: \t {}", loss,
			correct * 100);

		int batches = training.size() / batchSize;

		metrics.initialPlotData(correct, loss);

		for (int i = 1; i <= epochs; i++) {

			// Calculates a batch of training data and update the deltas.
			t1 = System.nanoTime();
			for (int j = 0; j <= batches; j++) {
				getBatch(j, batchSize, training).forEach(e -> {
					this.evaluate(e.getData(), e.getLabel());
				});
				this.fit();
			}
			t2 = System.nanoTime();

			// Feed forward the validation data
			Collections.shuffle(validation);
			final List<NetworkInput<M>> feedForwardData = this.feedforward(validation);

			// Evaluate prediction with the interface EvaluationFunction.
			correct = this.evaluationFunction.evaluatePrediction(feedForwardData);
			// Calculate cost/loss with the interface CostFunction
			loss = this.costFunction.calculateCostFunction(feedForwardData);

			// Add the plotting data, x, y_1, y_2 to the
			// lists of xValues, correctValues, lossValues.
			metrics.addPlotData(i, correct, loss, (t2 - t1));

			if ((i) % (epochs / 8) == 0) {
				log.info("\n {} / {} epochs are finished.\n Loss: \t {}\n Correct: \t {}", (i),
					epochs, loss,
					correct * 100);
			}
		}

		log.info("Outputting charts into " + path);
		try {
			metrics.present(path);
		} catch (final IOException e) {
			e.printStackTrace();
		}
		log.info("Charts outputted.");
	}

	@Override
	public double testEvaluation(final List<NetworkInput<M>> right, final int i) {
		double correct = 0d;
		var fedForward = feedforward(right);
		for (int j = 0; j < i; j++) {
			correct += this.evaluationFunction.evaluatePrediction(fedForward);
		}
		return correct / (double) i;
	}

	@Override
	public double testLoss(final List<NetworkInput<M>> right) {
		return this.costFunction.calculateCostFunction(feedforward(right));
	}

	@Override
	public void display() {
		final StringBuilder b = new StringBuilder();
		b.append("\n")
			.append("======================================================================")
			.append("\n")
			.append("Network information and structure.").append("\n")
			.append(String.format("Input nodes: [%d]; Output nodes: [%d]%n%n", this.inputNeurons,
				this.networkLayers.get(this.networkLayers.size() - 1).getNeurons()));

		for (int i = 0; i < this.networkLayers.size() - 1; i++) {
			var l = this.networkLayers.get(i);
			var l2 = this.networkLayers.get(i + 1);
			final int[] dims = new int[]{l.getNeurons(), l2.getNeurons()};
			b.append(String.format("\tLayer %d : [%d X %d]%n", (i + 1), dims[0], dims[1]))
				.append(String.format("\tActivation function from this layer: %s",
					l2.getFunction().getName()))
				.append("\n");
		}

		b.append("\n").append("The error function: ").append(this.costFunction.name()).append("\n")
			.append("The evaluation function: ").append(this.evaluationFunction.name()).append("\n")
			.append("The optimizer: ").append(this.optimizer.name()).append("\n")
			.append("======================================================================");

		log.info(b.toString());
	}

	@Override
	public void copyParameters(final List<Matrix<M>> weights, final List<Matrix<M>> biases) {

		for (int i = 0; i < weights.size(); i++) {
			int layerIndex = i + 1;
			this.networkLayers.get(layerIndex).setWeight(weights.get(i));
			this.networkLayers.get(layerIndex).setBias(biases.get(i));
		}

	}

	@Override
	public List<Pair<Matrix<M>, Matrix<M>>> getParameters() {
		return this.networkLayers.stream().filter(e -> e.getWeight() != null)
			.map(e -> Pair.of(e.getWeight(), e.getBias())).collect(toList());
	}

	public Matrix<M> predict(Matrix<M> input) {
		return this.checkEvaluate(input, null);
	}

	private Matrix<M> checkEvaluate(final Matrix<M> data, final Matrix<M> label) {
		if (label == null) {
			return this.evaluate(data, null);
		} else {
			return this.evaluate(data, label);
		}
	}

	private Matrix<M> evaluate(Matrix<M> input, Matrix<M> label) {
		Matrix<M> toEvaluate = input;
		for (var layer : networkLayers) {
			toEvaluate = layer.calculate(toEvaluate);
		}

		if (label != null) {
			this.backPropagation(label);
		}

		return toEvaluate;
	}

	private void backPropagation(Matrix<M> label) {
		var layer = getLastLayer();
		var lastActivation = layer.activation();

		var costDerivative = this.costFunction.applyCostFunctionGradient(lastActivation, label);

		do {
			// Also deltaBias.
			var dCdI = layer.getFunction().derivativeOnInput(lastActivation, costDerivative);

			Matrix<M> activation = layer.precedingLayer().activation().transpose();

			var deltaWeights = dCdI.multiply(activation);

			layer.addDeltas(deltaWeights, dCdI);

			costDerivative = layer.getWeight().transpose().multiply(dCdI);

			layer = layer.precedingLayer();
			lastActivation = layer.activation();

		} while (layer.hasPrecedingLayer());

	}

	private NetworkLayer<M> getLastLayer() {
		return networkLayers.get(networkLayers.size() - 1);
	}

	public synchronized void fit() {
		for (int i = 0; i < networkLayers.size(); i++) {
			var layer = networkLayers.get(i);
			if (layer.hasPrecedingLayer()) {
				layer.fit(i, this.optimizer);
			}
		}
	}

	public void train(List<NetworkInput<M>> training, List<NetworkInput<M>> validation,
		final int epochs,
		final int batchSize, boolean silent) {

		int batches = training.size() / batchSize;

		if (!silent) {
			log.info("Prior to training:");
			log.info("Initial loss: \t {}", this.testLoss(validation));
			log.info("Initial evaluation percentage: \t {}%.\n",
				this.testEvaluation(validation, 37) * 100);
		}
		for (int epoch = 0; epoch < epochs; epoch++) {
			if (!silent) {
				log.info("Epoch: \t {}", epoch + 1);
			}
			for (int i = 0; i <= batches; i++) {
				Collections.shuffle(training);
				getBatch(i, batchSize, training).forEach(e -> {
					this.evaluate(e.getData(), e.getLabel());
				});
				this.fit();
			}

			if (!silent) {
				if ((epoch + 1) % 5 == 0) {
					log.info("Loss: \t\t\t {}", this.testLoss(validation));
					log.info("Evaluation percentage: \t {}%.\n",
						this.testEvaluation(validation, 37) * 100);
				}
			}
		}
	}

	private List<NetworkInput<M>> feedforward(List<NetworkInput<M>> data) {
		return data.stream()
			.map(e -> new NetworkInput<M>(this.checkEvaluate(e.getData(), null), e.getLabel()))
			.collect(toList());
	}

	public Matrix<M> predict(Matrix<M> input, Matrix<M> label) {
		return this.checkEvaluate(input, label);
	}

	public int getInputSize() {
		return this.inputNeurons;
	}

	public CostFunction<M> getCostFunction() {
		return this.costFunction;
	}

	public Optimizer<M> getOptimizer() {
		return this.optimizer;
	}

	public List<NetworkLayer<M>> getLayers() {
		return this.networkLayers;
	}

	public ParameterInitializer<M> getInitializer() {
		return this.initializer;
	}

	public EvaluationFunction<M> getEvaluationFunction() {
		return this.evaluationFunction;
	}

	public boolean isClipping() {
		return clipping;
	}
}
