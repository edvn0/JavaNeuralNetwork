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

	public static final Map<String, CostFunction<Primitive64Matrix>> ojCostFunctions = Map.of("Cross Entropy",
			new CrossEntropyCostFunction<Primitive64Matrix>(), "Mean Squared Error",
			new MeanSquaredCostFunction<Primitive64Matrix>(), "Huber Loss",
			new SmoothL1CostFunction<Primitive64Matrix>());
	public static final Map<String, ActivationFunction<Primitive64Matrix>> ojFunctions = Map.of("DoNothing",
			new DoNothingFunction<Primitive64Matrix>(), "LeakyReLU", new LeakyReluFunction<Primitive64Matrix>(),
			"Linear", new LinearFunction<Primitive64Matrix>(), "ReLU", new ReluFunction<Primitive64Matrix>(), "Sigmoid",
			new SigmoidFunction<Primitive64Matrix>(), "Softmax", new SoftmaxFunction<Primitive64Matrix>(), "Tanh",
			new TanhFunction<Primitive64Matrix>());

	public static final Map<String, Optimizer<Primitive64Matrix>> ojOptimisers = Map.of("Adaptive Moment Estimation",
			new ADAM<Primitive64Matrix>(), "Stochastic Gradient Descent",
			new StochasticGradientDescent<Primitive64Matrix>(), "Momentum", new Momentum<Primitive64Matrix>());

	public static final Map<String, EvaluationFunction<Primitive64Matrix>> ojEvaluators = Map.of("Argmax Evaluation",
			new ArgMaxEvaluationFunction<Primitive64Matrix>(), "Threshold Evaluation",
			new ThresholdEvaluationFunction<Primitive64Matrix>());

	// UJMP

	public static final Map<String, ActivationFunction<org.ujmp.core.Matrix>> ujmpFunctions = Map.of("DoNothing",
			new DoNothingFunction<org.ujmp.core.Matrix>(), "LeakyReLU", new LeakyReluFunction<org.ujmp.core.Matrix>(),
			"Linear", new LinearFunction<org.ujmp.core.Matrix>(), "ReLU", new ReluFunction<org.ujmp.core.Matrix>(),
			"Sigmoid", new SigmoidFunction<org.ujmp.core.Matrix>(), "Softmax",
			new SoftmaxFunction<org.ujmp.core.Matrix>(), "Tanh", new TanhFunction<org.ujmp.core.Matrix>());

	public static final Map<String, Optimizer<org.ujmp.core.Matrix>> ujmpOptimisers = Map.of(
			"Adaptive Moment Estimation", new ADAM<org.ujmp.core.Matrix>(), "Stochastic Gradient Descent",
			new StochasticGradientDescent<org.ujmp.core.Matrix>(), "Momentum", new Momentum<org.ujmp.core.Matrix>());

	public static final Map<String, EvaluationFunction<org.ujmp.core.Matrix>> ujmpEvaluators = Map.of(
			"Argmax Evaluation", new ArgMaxEvaluationFunction<org.ujmp.core.Matrix>(), "Threshold Evaluation",
			new ThresholdEvaluationFunction<org.ujmp.core.Matrix>());

	public static final Map<String, CostFunction<org.ujmp.core.Matrix>> ujmpCostFunctions = Map.of("Cross Entropy",
			new CrossEntropyCostFunction<org.ujmp.core.Matrix>(), "Mean Squared Error",
			new MeanSquaredCostFunction<org.ujmp.core.Matrix>(), "Huber Loss",
			new SmoothL1CostFunction<org.ujmp.core.Matrix>());

	// SIMPLE

	public static final Map<String, ActivationFunction<SMatrix>> simpleFunctions = Map.of("DoNothing",
			new DoNothingFunction<SMatrix>(), "LeakyReLU", new LeakyReluFunction<SMatrix>(), "Linear",
			new LinearFunction<SMatrix>(), "ReLU", new ReluFunction<SMatrix>(), "Sigmoid",
			new SigmoidFunction<SMatrix>(), "Softmax", new SoftmaxFunction<SMatrix>(), "Tanh",
			new TanhFunction<SMatrix>());

	public static final Map<String, Optimizer<SMatrix>> simpleOptimisers = Map.of("Adaptive Moment Estimation",
			new ADAM<SMatrix>(), "Stochastic Gradient Descent", new StochasticGradientDescent<SMatrix>(), "Momentum",
			new Momentum<SMatrix>());

	public static final Map<String, EvaluationFunction<SMatrix>> simpleEvaluators = Map.of("Argmax Evaluation",
			new ArgMaxEvaluationFunction<SMatrix>(), "Threshold Evaluation",
			new ThresholdEvaluationFunction<SMatrix>());

	public static final Map<String, CostFunction<SMatrix>> simpleCostFunctions = Map.of("Cross Entropy",
			new CrossEntropyCostFunction<SMatrix>(), "Mean Squared Error", new MeanSquaredCostFunction<SMatrix>(),
			"Huber Loss", new SmoothL1CostFunction<SMatrix>());

}