package reinforcement.learning.dqn;

import static java.util.stream.Collectors.toList;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;
import lombok.Getter;
import math.activations.LinearFunction;
import math.activations.ReluFunction;
import math.costfunctions.MeanSquaredCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.optimizers.ADAM;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.OjAlgoInitializer;
import neuralnetwork.inputs.NetworkInput;
import neuralnetwork.layer.LayeredNetworkBuilder;
import neuralnetwork.layer.LayeredNeuralNetwork;
import neuralnetwork.layer.NetworkLayer;
import org.ojalgo.matrix.Primitive64Matrix;
import reinforcement.env.BaseEnvironment;
import reinforcement.learning.agent.LearningAgent;
import reinforcement.learning.er.Transition;
import reinforcement.utils.EnvObservation;
import utilities.types.Pair;

public class DQNAgent<ObsT> extends LearningAgent<ObsT> {

	private static final double EPSILON_DECREMENT_FACTOR = 1 - 1e-4;
	private static final double EPSILON_MINIMUM = 0.01;
	private static final int MAX_STEPS = 50;
	private final int actionSize;
	private double gamma = 0.99;
	private LayeredNeuralNetwork<Primitive64Matrix> policy;
	private LayeredNeuralNetwork<Primitive64Matrix> target;

	@Getter
	private double epsilon;

	private int learningSteps;

	public DQNAgent(final BaseEnvironment<Integer, ObsT> env, double lR, double gamma,
		int hiddenNodes) {
		super(env);
		int inputs = env.getObservationSpace().shape().getX();
		int outputs = env.getActionSpace().shape().getX();
		actionSize = outputs;
		var builder = new LayeredNetworkBuilder<Primitive64Matrix>()
			.optimizer(new ADAM<>(lR, 0.9, 0.999))
			.costFunction(new MeanSquaredCostFunction<>())
			.evaluationFunction(new ArgMaxEvaluationFunction<>())
			.initializer(new OjAlgoInitializer(MethodConstants.XAVIER, MethodConstants.XAVIER))
			.layer(new NetworkLayer<>(new ReluFunction<>(), inputs))
			.layer(new NetworkLayer<>(new ReluFunction<>(), hiddenNodes))
			.layer(new NetworkLayer<>(new ReluFunction<>(), hiddenNodes))
			.layer(new NetworkLayer<>(new LinearFunction<>(1), outputs));

		this.policy = builder.create();
		this.target = builder.create();

		this.learningSteps = 0;

		this.epsilon = 0.99;
		this.gamma = gamma;
	}

	/**
	 * Deserializing constructor.
	 *
	 * @param serializePath where is the agent serialized?
	 */
	public DQNAgent(final File serializePath,
		final BaseEnvironment<Integer, ObsT> env) {
		super(serializePath, env);
		actionSize = env.getActionSpace().shape().getX();
	}

	@Override
	public void updateParameters() {
		var params = this.policy.getParameters();
		var ws = params.stream().map(Pair::left).collect(toList());
		var bs = params.stream().map(Pair::right).collect(toList());
		this.target.copyParameters(ws, bs);
	}

	@Override
	public void serialize(final String outputPath) {

	}

	@Override
	public Integer act(final EnvObservation observation) {
		if (this.getRandom().nextDouble() < this.epsilon) {
			this.epsilon *= EPSILON_DECREMENT_FACTOR;
			this.epsilon = Math.max(this.epsilon, EPSILON_MINIMUM);
			int randomMove = this.getRandom().nextInt(actionSize);
			return randomMove;
		} else {
			var state = new OjAlgoMatrix(observation.getObservations());
			var pred = this.policy.predict(state);
			return pred.argMax();
		}
	}

	@Override
	public void learn(final List<Transition> transitions) {
		learningSteps++;
		this.learningSteps = this.learningSteps % MAX_STEPS;

		if (this.learningSteps == 0) {
			this.updateParameters();
		}

		if (transitions.isEmpty()) {
			return;
		}

		var bellman = this.toBellman(transitions);
		this.policy.train(bellman, 1, 64);
	}

	private List<NetworkInput<Primitive64Matrix>> toBellman(final List<Transition> sample) {
		return sample.parallelStream().map(transition -> {
			var oldState = new OjAlgoMatrix(transition.getS().getObservations());
			var oldStateCopy = oldState.copy();
			var newState = new OjAlgoMatrix(transition.getNewS().getObservations());
			var done = transition.isDone() ? 0 : 1;
			var reward = transition.getReward();
			var action = transition.getAction();

			var oldQ = this.policy.predict(oldState.copy()).rawCopy();
			var newQ = this.target.predict(newState);

			var max = newQ.max();

			oldQ[action][0] = max * gamma * done + reward;

			return new NetworkInput<>(oldStateCopy, new OjAlgoMatrix(oldQ));
		}).collect(Collectors.toList());
	}

	@Override
	public void setIsTraining(final boolean isTraining) {
		super.setIsTraining(isTraining);
		this.epsilon = EPSILON_MINIMUM;
	}


}
