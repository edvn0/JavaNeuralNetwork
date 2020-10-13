package neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.initialiser.ParameterInitialiser;
import neuralnetwork.inputs.NetworkInput;
import neuralnetwork.layer.NetworkLayer;

@Slf4j
public class LayeredNeuralNetwork<M> implements DeepLearnable<M> {

	private final List<NetworkLayer<M>> networkLayers;
	// The error function to minimize.
	private final CostFunction<M> costFunction;
	// The function to evaluate the data set.
	private final EvaluationFunction<M> evaluationFunction;
	// The optimizer to be used
	private final Optimizer<M> optimizer;

	private final ParameterInitialiser<M> initializer;

	public LayeredNeuralNetwork(LayeredNetworkBuilder<M> b) {
		this.costFunction = b.costFunction;
		this.evaluationFunction = b.evaluationFunction;

		this.optimizer = b.optimizer;
		this.optimizer.initializeOptimizer(b.total, null, null);

		this.initializer = b.initializer;
		initializer.init(b.calculateStructure());

		this.networkLayers = new ArrayList<>();

		var first = b.layers.get(0);
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

			layer.setPrecedingLayer(prev);
			this.networkLayers.add(layer);

			prev = layer;
		}
	}

	public LayeredNeuralNetwork(LayeredNetworkBuilder<M> b, boolean deser) {
		this.costFunction = b.costFunction;
		this.evaluationFunction = b.evaluationFunction;

		this.optimizer = b.optimizer;
		this.optimizer.initializeOptimizer(b.total, null, null);

		this.initializer = b.initializer;
		initializer.init(b.calculateStructure());

		this.networkLayers = new ArrayList<>();

		var first = b.layers.get(0);
		NetworkLayer<M> firstLayer = new NetworkLayer<>(first.getFunction(), first.getNeurons());
		this.networkLayers.add(firstLayer);

		for (int i = 1; i < b.total; i++) {
			var layer = new NetworkLayer<>(b.layers.get(i));
			this.networkLayers.add(layer);
		}
	}

	private static <U> List<U> getBatch(final int i, final int batchSize,
		final List<U> data) {
		int fromIx = i * batchSize;
		int toIx = Math.min(data.size(), (i + 1) * batchSize);
		return Collections.unmodifiableList(data.subList(fromIx, toIx));
	}

	public void train(List<NetworkInput<M>> training, List<NetworkInput<M>> validation,
		final int epochs,
		final int batchSize) {

		int batches = training.size() / batchSize;

		log.info("Prior to training:");
		log.info("Initial loss: \t {}", this.testLoss(validation));
		log.info("Initial evaluation percentage: \t {}%.\n",
			this.testEvaluation(validation, 37) * 100);

		for (int epoch = 0; epoch < epochs; epoch++) {
			log.info("Epoch: \t {}", epoch + 1);
			for (int i = 0; i <= batches; i++) {
				Collections.shuffle(training);
				getBatch(i, batchSize, training).parallelStream()
					.forEach(e -> {
						this.evaluate(e.getData(), e.getLabel());
					});
				this.fit();
			}

			if ((epoch + 1) % 5 == 0) {
				log.info("Loss: \t\t\t {}", this.testLoss(validation));
				log.info("Evaluation percentage: \t {}%.\n",
					this.testEvaluation(validation, 37) * 100);
			}
		}
	}

	@Override
	public void trainWithMetrics(final List<NetworkInput<M>> training,
		final List<NetworkInput<M>> validation,
		final int epochs, final int batchSize, final String outputPath) {

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
		for (var l : networkLayers) {
			System.out.println(l);
		}
	}

	private List<NetworkInput<M>> feedforward(List<NetworkInput<M>> data) {
		return data.stream()
			.map(e -> new NetworkInput<M>(
				this.checkEvaluate(e.getData(), null),
				e.getLabel()))
			.collect(Collectors.toList());
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


	public Matrix<M> predict(Matrix<M> input) {
		return this.checkEvaluate(input, null);
	}

	public Matrix<M> predict(Matrix<M> input, Matrix<M> label) {
		return this.checkEvaluate(input, label);
	}

	private void backPropagation(Matrix<M> label) {
		var layer = getLastLayer();
		var lastActivation = layer.activation();

		var costDerivative = this.costFunction
			.applyCostFunctionGradient(lastActivation, label);

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

	public int getInputSize() {
		return this.networkLayers.get(0).getNeurons();
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

	public ParameterInitialiser<M> getInitializer() {
		return this.initializer;
	}

	public EvaluationFunction<M> getEvaluationFunction() {
		return this.evaluationFunction;
	}
}
