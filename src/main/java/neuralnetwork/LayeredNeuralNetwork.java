package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.initialiser.ParameterInitialiser;
import neuralnetwork.layer.NetworkLayer;
import neuralnetwork.layer.ZVector;

public class LayeredNeuralNetwork<M> {

	private List<NetworkLayer<M>> networkLayers;
	// The error function to minimize.
	private final CostFunction<M> costFunction;
	// The function to evaluate the data set.
	private final EvaluationFunction<M> evaluationFunction;
	// The optimizer to be used
	private final Optimizer<M> optimizer;

	private final ParameterInitialiser<M> initialiser;

	public LayeredNeuralNetwork(LayeredNetworkBuilder<M> b) {
		this.costFunction = b.costFunction;
		this.evaluationFunction = b.evaluationFunction;

		System.out.println(b.layers);

		this.optimizer = b.optimizer;
		this.optimizer.initializeOptimizer(b.total, null, null);

		this.initialiser = b.initializer;
		initialiser.init(b.calculateStructure());

		this.networkLayers = new ArrayList<>();

		NetworkLayer<M> firstLayer = new NetworkLayer<>(b.layers.get(0));
		this.networkLayers.add(firstLayer);

		var weights = initialiser.getWeightParameters();
		var biases = initialiser.getBiasParameters();

		var prev = firstLayer;
		for (int i = 1; i < b.total; i++) {
			var layer = new NetworkLayer<>(b.layers.get(i));
			var layerWeight = weights.get(i - 1);
			var layerBias = biases.get(i - 1);

			layer.setWeight(layerWeight);
			layer.setBias(layerBias);
			layer.setPrecedingLayer(prev);

			this.networkLayers.add(layer);
		}

		this.networkLayers = networkLayers.subList(1, networkLayers.size());
	}

	public ZVector<M> predict(Matrix<M> input) {
		return this.evaluate(new ZVector<>(input), null);
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

			var deltaWeights = dCdI.multiply(layer.precedingLayer().activation());

			layer.addDeltas(deltaWeights, dCdI);

			layer = layer.precedingLayer();

		} while (layer.hasPrecedingLayer());

	}

	public synchronized void fit() {
		for (int i = 0; i < networkLayers.size(); i++) {
			var layer = networkLayers.get(i);
			if (layer.hasPrecedingLayer()) {
				layer.fit(i, this.optimizer);
			}
		}
	}

	private NetworkLayer<M> getLastLayer() {
		return networkLayers.get(networkLayers.size() - 1);
	}

	public int getInputSize() {
		return 0;
	}

	public CostFunction<M> getCostFunction() {
		return null;
	}

	public Optimizer<M> getOptimizer() {
		return null;
	}

	public List<NetworkLayer<M>> getLayers() {
		return null;
	}

}
