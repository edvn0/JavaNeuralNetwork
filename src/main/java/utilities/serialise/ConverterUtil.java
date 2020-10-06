package utilities.serialise;

import java.util.Map;

import org.ojalgo.matrix.Primitive64Matrix;

import math.activations.ActivationFunction;
import math.activations.DoNothingFunction;
import math.activations.LeakyReluFunction;
import math.activations.LinearFunction;
import math.activations.ReluFunction;
import math.activations.SigmoidFunction;
import math.activations.SoftmaxFunction;
import math.activations.TanhFunction;
import math.costfunctions.BinaryCrossEntropyCostFunction;
import math.costfunctions.CostFunction;
import math.costfunctions.CrossEntropyCostFunction;
import math.costfunctions.MeanSquaredCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.ThresholdEvaluationFunction;
import math.optimizers.ADAM;
import math.optimizers.Momentum;
import math.optimizers.Optimizer;
import math.optimizers.StochasticGradientDescent;

public final class ConverterUtil {

        public static final Map<String, ActivationFunction<Primitive64Matrix>> ojFunctions = Map.of("DoNothing",
                        new DoNothingFunction<Primitive64Matrix>(), "LeakyReLU",
                        new LeakyReluFunction<Primitive64Matrix>(), "Linear", new LinearFunction<Primitive64Matrix>(),
                        "ReLU", new ReluFunction<Primitive64Matrix>(), "Sigmoid",
                        new SigmoidFunction<Primitive64Matrix>(), "Softmax", new SoftmaxFunction<Primitive64Matrix>(),
                        "Tanh", new TanhFunction<Primitive64Matrix>());

        public static final Map<String, ActivationFunction<org.ujmp.core.Matrix>> ujmpFunctions = Map.of("DoNothing",
                        new DoNothingFunction<org.ujmp.core.Matrix>(), "LeakyReLU",
                        new LeakyReluFunction<org.ujmp.core.Matrix>(), "Linear",
                        new LinearFunction<org.ujmp.core.Matrix>(), "ReLU", new ReluFunction<org.ujmp.core.Matrix>(),
                        "Sigmoid", new SigmoidFunction<org.ujmp.core.Matrix>(), "Softmax",
                        new SoftmaxFunction<org.ujmp.core.Matrix>(), "Tanh", new TanhFunction<org.ujmp.core.Matrix>());

        public static final Map<String, Optimizer<Primitive64Matrix>> ojOptimisers = Map.of(
                        "Adaptive Moment Estimation", new ADAM<Primitive64Matrix>(), "Stochastic Gradient Descent",
                        new StochasticGradientDescent<Primitive64Matrix>(), "Momentum",
                        new Momentum<Primitive64Matrix>());

        public static final Map<String, Optimizer<org.ujmp.core.Matrix>> ujmpOptimisers = Map.of(
                        "Adaptive Moment Estimation", new ADAM<org.ujmp.core.Matrix>(), "Stochastic Gradient Descent",
                        new StochasticGradientDescent<org.ujmp.core.Matrix>(), "Momentum",
                        new Momentum<org.ujmp.core.Matrix>());

        public static final Map<String, EvaluationFunction<Primitive64Matrix>> ojEvaluators = Map.of(
                        "Argmax Evaluation", new ArgMaxEvaluationFunction<Primitive64Matrix>(), "Threshold Evaluation",
                        new ThresholdEvaluationFunction<Primitive64Matrix>());

        public static final Map<String, EvaluationFunction<org.ujmp.core.Matrix>> ujmpEvaluators = Map.of(
                        "Argmax Evaluation", new ArgMaxEvaluationFunction<org.ujmp.core.Matrix>(),
                        "Threshold Evaluation", new ThresholdEvaluationFunction<org.ujmp.core.Matrix>());

        public static final Map<String, CostFunction<Primitive64Matrix>> ojCostFunctions = Map.of(
                        "Binary Cross Entropy", new BinaryCrossEntropyCostFunction<Primitive64Matrix>(),
                        "Cross Entropy", new CrossEntropyCostFunction<Primitive64Matrix>(), "Mean Squared Error",
                        new MeanSquaredCostFunction<Primitive64Matrix>());

        public static final Map<String, CostFunction<org.ujmp.core.Matrix>> ujmpCostFunctions = Map.of(
                        "Binary Cross Entropy", new BinaryCrossEntropyCostFunction<org.ujmp.core.Matrix>(),
                        "Cross Entropy", new CrossEntropyCostFunction<org.ujmp.core.Matrix>(), "Mean Squared Error",
                        new MeanSquaredCostFunction<org.ujmp.core.Matrix>());
}
