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
import neuralnetwork.layer.ZVector;
import org.jetbrains.annotations.NotNull;
import utilities.types.Pair;

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

	public void train(List<NetworkInput<M>> training, List<NetworkInput<M>> validation,
		final int epochs,
		final int batchSize) {

		var zVectorTraining = training.stream()
			.map(e -> Pair.of(new ZVector<>(e.getData()), new ZVector<>(e.getLabel()))).collect(
				Collectors.toList());
		int batches = training.size() / batchSize;
		for (int i = 0; i < batches; i++) {
			double trainingEvaluation = getBatch(i, batchSize, zVectorTraining).parallelStream()
				.map(e ->
					Pair.of(this.evaluate(e.left(), e.right()), e.right())
				).mapToDouble(e -> this.evaluationFunction
					.evaluateSingle(NetworkInput.of(e.left(), e.right()))).reduce(Double::sum)
				.orElse(0d) / batchSize;

			log.info("Evaluation percentage: \t {}%.", trainingEvaluation * 100);

			this.fit();
		}
	}

	@Override
	public void trainWithMetrics(final List<NetworkInput<M>> left,
		final List<NetworkInput<M>> middle,
		final int left1, final int right, final String outputPath) {

	}

	@Override
	public double testEvaluation(final List<NetworkInput<M>> right, final int i) {
		double correct = 0d;
		var fedForward = right.stream().map(
			e -> new NetworkInput<>(this.evaluate(e.getData(), null).getZVector(),
				e.getLabel())).collect(Collectors.toList());
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
				this.evaluate(new ZVector<>(e.getData()), new ZVector<M>(e.getLabel()))
					.getZVector(),
				e.getLabel()))
			.collect(Collectors.toList());
	}

	private ZVector<M> evaluate(final Matrix<M> data, final Matrix<M> label) {
		ZVector<M> input = new ZVector<M>(data);
		if (label != null) {
			ZVector<M> correct = new ZVector<M>(label);
			return evaluate(input, correct);
		}
		return evaluate(input, null);

	}

	private <U> List<U> getBatch(final int i, final int batchSize,
		final List<U> data) {
		int fromIx = i * batchSize;
		int toIx = Math.min(data.size(), (i + 1) * batchSize);
		return Collections.unmodifiableList(data.subList(fromIx, toIx));
	}

	private ZVector<M> evaluate(ZVector<M> input, ZVector<M> label) {
		ZVector<M> toEvaluate = new ZVector<>(input);
		for (var layer : networkLayers) {
			toEvaluate = layer.calculate(toEvaluate);
		}

		if (label != null) {
			this.backPropagation(label);
		}

		return toEvaluate;
	}

	private void backPropagation(ZVector<M> label) {
		var layer = getLastLayer();

		var costDerivative = this.costFunction
			.applyCostFunctionGradient(layer.activation(), label.getZVector());

		do {
			// Also deltaBias.
			var dCdI = layer.getFunction().derivativeOnInput(layer.activation(), costDerivative);

			Matrix<M> activation = layer.precedingLayer().activation().transpose();

			var deltaWeights = dCdI.multiply(activation);

			layer.addDeltas(deltaWeights, dCdI);

			costDerivative = layer.getWeight().transpose().multiply(dCdI);

			layer = layer.precedingLayer();

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

	public double feedforwardEvaluation(final List<Pair<ZVector<M>, ZVector<M>>> unfedTestingData) {
		List<NetworkInput<M>> l = feedforwardZVectors(unfedTestingData);

		double correct = 0d;
		for (int i = 0; i < 100; i++) {
			correct += this.evaluationFunction.evaluatePrediction(l);
		}

		return correct / 100;
	}

	@NotNull
	private List<NetworkInput<M>> feedforwardZVectors(
		final List<Pair<ZVector<M>, ZVector<M>>> unfedTestingData) {
		return unfedTestingData.stream()
			.map(e -> new NetworkInput<M>(this.evaluate(e.left(), e.right()).getZVector(),
				e.right().getZVector()))
			.collect(Collectors.toList());
	}

	public ZVector<M> predict(Matrix<M> input) {
		return this.evaluate(new ZVector<>(input), null);
	}

	public ZVector<M> predict(Matrix<M> input, Matrix<M> label) {
		return this.evaluate(new ZVector<>(input), new ZVector<>(label));
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
