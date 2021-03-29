package reinforcement.learning.agent;

import java.io.File;
import java.util.List;
import reinforcement.env.BaseEnvironment;
import reinforcement.learning.er.Transition;
import reinforcement.utils.EnvObservation;

public abstract class LearningAgent<ObsT> {

	private final BaseEnvironment<Integer, ObsT> env;
	private boolean isTraining;

	public LearningAgent(BaseEnvironment<Integer, ObsT> env) {
		this.env = env;
		this.isTraining = true;
	}

	/**
	 * Deserializing constructor. By default, disables training.
	 *
	 * @param serializePath where is the agent serialized?
	 */
	public LearningAgent(File serializePath, BaseEnvironment<Integer, ObsT> env) {
		this.env = env;
		this.isTraining = false;
	}

	public abstract void updateParameters();

	public abstract void serialize(final String outputPath);

	public abstract Integer act(final EnvObservation observation);

	public abstract void learn(List<Transition> transitions);

	public void setIsTraining(final boolean isTraining) {
		this.isTraining = isTraining;
	}

	public boolean isTraining() {
		return this.isTraining;
	}
}
