package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import math.costfunctions.CostFunction;
import math.costfunctions.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.evaluation.EvaluationFunction;
import math.optimizers.ADAM;
import math.optimizers.Optimizer;
import neuralnetwork.initialiser.ParameterInitialiser;
import neuralnetwork.layer.NetworkLayer;

public class LayeredNetworkBuilder<M> {

	public List<NetworkLayer<M>> layers = new ArrayList<>();
	public int total;
	public int networkInputSize;

	public ParameterInitialiser<M> initializer;
	public CostFunction<M> costFunction = new CrossEntropyCostFunction<>();
	public EvaluationFunction<M> evaluationFunction = new ArgMaxEvaluationFunction<>();
	public Optimizer<M> optimizer = new ADAM<>(0.01, 0.9, 0.999);

	public LayeredNetworkBuilder(int networkInputSize) {
		this.networkInputSize = networkInputSize;
	}

	public LayeredNetworkBuilder<M> evaluationFunction(EvaluationFunction<M> evaluationFunction) {
		this.evaluationFunction = evaluationFunction;
		return this;
	}

	public LayeredNetworkBuilder<M> initialiser(ParameterInitialiser<M> initialiser) {
		this.initializer = initialiser;
		return this;
	}

	public LayeredNetworkBuilder<M> costFunction(CostFunction<M> costFunction) {
		this.costFunction = costFunction;
		return this;
	}

	public LayeredNetworkBuilder<M> optimizer(Optimizer<M> optimizer) {
		this.optimizer = optimizer;
		return this;
	}

	public LayeredNetworkBuilder<M> layer(NetworkLayer<M> layer) {
		layers.add(layer);
		return this;
	}

	public LayeredNeuralNetwork<M> create() {
		this.total = this.layers.size();
		return new LayeredNeuralNetwork<>(this);
	}

	public int[] calculateStructure() {
		int[] structure = new int[this.total];
		for (int i = 0; i < this.total; i++) {
			int prevSize = this.layers.get(i).getNeurons();
			structure[i] = prevSize;
		}
		return structure;
	}

}
