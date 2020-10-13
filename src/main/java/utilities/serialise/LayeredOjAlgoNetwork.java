package utilities.serialise;

import java.util.List;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.optimizers.Optimizer;
import neuralnetwork.LayeredNetworkBuilder;
import neuralnetwork.LayeredNeuralNetwork;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.OjAlgoInitialiser;
import neuralnetwork.layer.NetworkLayer;
import org.ojalgo.matrix.Primitive64Matrix;

public class LayeredOjAlgoNetwork {

	public static LayeredNeuralNetwork<Primitive64Matrix> create(int inputSize,
		List<NetworkLayer<Primitive64Matrix>> layers,
		CostFunction<Primitive64Matrix> costFunc,
		Optimizer<Primitive64Matrix> optimiser, EvaluationFunction<Primitive64Matrix> evaluator) {

		LayeredNetworkBuilder<Primitive64Matrix> builder = new LayeredNetworkBuilder<>(inputSize);
		builder = builder.costFunction(costFunc);
		builder = builder.evaluationFunction(evaluator);
		builder = builder.optimizer(optimiser);

		for (var l : layers) {
			System.out.println(l);
			builder = builder.layer(l);
		}

		OjAlgoInitialiser initializer = new OjAlgoInitialiser(MethodConstants.XAVIER,
			MethodConstants.SCALAR);
		initializer.init(builder.calculateStructure());

		builder = builder.initializer(initializer);

		return builder.deserialize();
	}

}
