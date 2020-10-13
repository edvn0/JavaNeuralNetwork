package utilities.serialise;

import java.util.List;
import math.activations.ActivationFunction;
import math.costfunctions.CostFunction;
import math.evaluation.EvaluationFunction;
import math.linearalgebra.Matrix;
import math.optimizers.Optimizer;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.UJMPInitialiser;

public class UJMPNetwork {

	public static NeuralNetwork<org.ujmp.core.Matrix> create(
		List<Matrix<org.ujmp.core.Matrix>> weights,
		List<Matrix<org.ujmp.core.Matrix>> biases, int layers, int[] sizes,
		List<ActivationFunction<org.ujmp.core.Matrix>> functions,
		CostFunction<org.ujmp.core.Matrix> costFunc,
		Optimizer<org.ujmp.core.Matrix> optimiser,
		EvaluationFunction<org.ujmp.core.Matrix> evaluator) {

		NetworkBuilder<org.ujmp.core.Matrix> builder = new NetworkBuilder<>(layers);
		builder.setCostFunction(costFunc);
		builder.setEvaluationFunction(evaluator);
		builder.setOptimizer(optimiser);

		builder.setFirstLayer(sizes[0]);

		int[] paramSizes = new int[sizes.length - 1];
		for (int i = 1; i < sizes.length - 1; i++) {
			builder.setLayer(sizes[i], functions.get(i));
			paramSizes[i - 1] = sizes[i];
		}
		paramSizes[paramSizes.length - 1] = sizes[sizes.length - 1];

		builder.setLastLayer(sizes[sizes.length - 1], functions.get(functions.size() - 1));

		builder.setWeights(weights);
		builder.setBiases(biases);

		UJMPInitialiser initialiser = new UJMPInitialiser(MethodConstants.XAVIER,
			MethodConstants.SCALAR);
		initialiser.init(paramSizes);

		NeuralNetwork<org.ujmp.core.Matrix> out = new NeuralNetwork<>(builder, initialiser);

		return out;
	}

}
