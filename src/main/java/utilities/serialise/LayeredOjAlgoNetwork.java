package utilities.serialise;

import java.util.List;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.optimizers.Optimizer;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.OjAlgoInitializer;
import neuralnetwork.layer.LayeredNetworkBuilder;
import neuralnetwork.layer.LayeredNeuralNetwork;
import neuralnetwork.layer.NetworkLayer;
import org.ojalgo.matrix.Primitive64Matrix;

public class LayeredOjAlgoNetwork {

	public static LayeredNeuralNetwork<Primitive64Matrix> create(int inputSize,
		List<NetworkLayer<Primitive64Matrix>> layers,
		CostFunction<Primitive64Matrix> costFunc,
		Optimizer<Primitive64Matrix> optimiser, EvaluationFunction<Primitive64Matrix> evaluator) {

		LayeredNetworkBuilder<Primitive64Matrix> builder = new LayeredNetworkBuilder<>();
		builder = builder.costFunction(costFunc);
		builder = builder.evaluationFunction(evaluator);
		builder = builder.optimizer(optimiser);

		for (var l : layers) {
			builder = builder.layer(l);
		}

		OjAlgoInitializer initializer = new OjAlgoInitializer(MethodConstants.XAVIER,
			MethodConstants.SCALAR);
		initializer.init(builder.calculateStructure());

		builder = builder.initializer(initializer);

		return builder.deserialize();
	}

}
