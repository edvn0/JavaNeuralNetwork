package reinforcement.env;

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import lombok.Data;
import reinforcement.env.renderer.EnvRenderer;
import reinforcement.env.space.RewardRange;
import reinforcement.env.space.Space;
import reinforcement.utils.Sars;

@Data
public abstract class BaseEnvironment<ActionT, ObsT> implements AutoCloseable {

	protected SecureRandom random;
	protected Space<ActionT> actionSpace;
	protected Space<ObsT> observationSpace;
	protected RewardRange rewardRange;
	protected EnvRenderer<ActionT, ObsT> renderer;

	public BaseEnvironment() {
		try {
			this.random = SecureRandom.getInstanceStrong();
		} catch (NoSuchAlgorithmException e) {
			this.random = new SecureRandom();
		}
	}

	public abstract Sars step(ActionT a);

	public abstract BaseEnvironment<ActionT, ObsT> reset();

	public void render() {
		this.renderer.render();
	}

	public abstract void seed(long seed);

	public abstract void close() throws Exception;
}
