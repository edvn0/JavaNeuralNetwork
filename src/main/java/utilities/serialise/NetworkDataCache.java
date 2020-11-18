package utilities.serialise;

import java.util.Map;
import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;
import math.activations.LeakyReluFunction;
import math.activations.LinearFunction;
import math.activations.ReluFunction;
import math.activations.SigmoidFunction;
import math.activations.SoftmaxFunction;
import math.activations.TanhFunction;
import math.costfunctions.CostFunction;
import math.costfunctions.CrossEntropyCostFunction;
import math.costfunctions.MeanSquaredCostFunction;
import math.costfunctions.SmoothL1CostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.ThresholdEvaluationFunction;
import math.linearalgebra.simple.SMatrix;
import math.optimizers.ADAM;
import math.optimizers.Momentum;
import math.optimizers.Optimizer;
import math.optimizers.StochasticGradientDescent;
import org.ojalgo.matrix.Primitive64Matrix;

public final class NetworkDataCache {

	// OJALGO

	public static final Map<String, CostFunction<Primitive64Matrix>> ojCostFunctions = Map
		.of("Cross Entropy",
			new CrossEntropyCostFunction<>(), "Mean Squared Error",
			new MeanSquaredCostFunction<>(), "Huber Loss",
			new SmoothL1CostFunction<>());
	public static final Map<String, ActivationFunction<Primitive64Matrix>> ojFunctions = Map
		.of("DoNothing",
			new DoNothingFunction<>(), "LeakyReLU", new LeakyReluFunction<>(),
			"Linear", new LinearFunction<>(), "ReLU", new ReluFunction<>(), "Sigmoid",
			new SigmoidFunction<>(), "Softmax", new SoftmaxFunction<>(), "Tanh",
			new TanhFunction<>());

	public static final Map<String, Optimizer<Primitive64Matrix>> ojOptimisers = Map
		.of("Adaptive Moment Estimation",
			new ADAM<>(), "Stochastic Gradient Descent",
			new StochasticGradientDescent<>(), "Momentum", new Momentum<>());

	public static final Map<String, EvaluationFunction<Primitive64Matrix>> ojEvaluators = Map
		.of("Argmax Evaluation",
			new ArgMaxEvaluationFunction<>(), "Threshold Evaluation",
			new ThresholdEvaluationFunction<>());

	// UJMP

	public static final Map<String, ActivationFunction<org.ujmp.core.Matrix>> ujmpFunctions = Map
		.of("DoNothing",
			new DoNothingFunction<>(), "LeakyReLU", new LeakyReluFunction<>(),
			"Linear", new LinearFunction<>(), "ReLU", new ReluFunction<>(),
			"Sigmoid", new SigmoidFunction<>(), "Softmax",
			new SoftmaxFunction<>(), "Tanh", new TanhFunction<>());

	public static final Map<String, Optimizer<org.ujmp.core.Matrix>> ujmpOptimisers = Map.of(
		"Adaptive Moment Estimation", new ADAM<>(), "Stochastic Gradient Descent",
		new StochasticGradientDescent<>(), "Momentum", new Momentum<>());

	public static final Map<String, EvaluationFunction<org.ujmp.core.Matrix>> ujmpEvaluators = Map
		.of(
			"Argmax Evaluation", new ArgMaxEvaluationFunction<>(), "Threshold Evaluation",
			new ThresholdEvaluationFunction<>());

	public static final Map<String, CostFunction<org.ujmp.core.Matrix>> ujmpCostFunctions = Map
		.of("Cross Entropy",
			new CrossEntropyCostFunction<>(), "Mean Squared Error",
			new MeanSquaredCostFunction<>(), "Huber Loss",
			new SmoothL1CostFunction<>());

	// SIMPLE

	public static final Map<String, ActivationFunction<SMatrix>> simpleFunctions = Map
		.of("DoNothing",
			new DoNothingFunction<>(), "LeakyReLU", new LeakyReluFunction<>(), "Linear",
			new LinearFunction<>(), "ReLU", new ReluFunction<>(), "Sigmoid",
			new SigmoidFunction<>(), "Softmax", new SoftmaxFunction<>(), "Tanh",
			new TanhFunction<>());

	public static final Map<String, Optimizer<SMatrix>> simpleOptimisers = Map
		.of("Adaptive Moment Estimation",
			new ADAM<>(), "Stochastic Gradient Descent", new StochasticGradientDescent<>(),
			"Momentum",
			new Momentum<>());

	public static final Map<String, EvaluationFunction<SMatrix>> simpleEvaluators = Map
		.of("Argmax Evaluation",
			new ArgMaxEvaluationFunction<>(), "Threshold Evaluation",
			new ThresholdEvaluationFunction<>());

	public static final Map<String, CostFunction<SMatrix>> simpleCostFunctions = Map
		.of("Cross Entropy",
			new CrossEntropyCostFunction<>(), "Mean Squared Error", new MeanSquaredCostFunction<>(),
			"Huber Loss", new SmoothL1CostFunction<>());

}
