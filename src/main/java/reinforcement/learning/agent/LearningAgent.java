package reinforcement.learning.agent;

import java.io.File;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.List;
import java.util.Random;
import lombok.Getter;
import reinforcement.env.BaseEnvironment;
import reinforcement.learning.er.Transition;
import reinforcement.utils.EnvObservation;

public abstract class LearningAgent<ObsT> {

	protected final BaseEnvironment<Integer, ObsT> env;

	@Getter
	protected Random random;

	private boolean isTraining;

	public LearningAgent(BaseEnvironment<Integer, ObsT> env) {
		this.env = env;
		this.isTraining = true;
		try {
			this.random = SecureRandom.getInstanceStrong();
		} catch (NoSuchAlgorithmException e) {
			this.random = new SecureRandom();
		}
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
