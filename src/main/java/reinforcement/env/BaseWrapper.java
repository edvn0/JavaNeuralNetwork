package reinforcement.env;

import reinforcement.env.actions.EnvironmentAction;
import reinforcement.env.renderer.EnvRenderer;
import reinforcement.utils.Sars;

public abstract class BaseWrapper<T, U> extends BaseEnvironment<T, U> {

	private BaseEnvironment<T, U> env;

	public BaseWrapper(BaseEnvironment<T, U> env) {
		this.random = env.random;
		this.rewardRange = env.rewardRange;
		this.actionSpace = env.actionSpace;
		this.observationSpace = env.observationSpace;
	}

	public abstract Sars step(EnvironmentAction a);

	public abstract BaseEnvironment<T, U> reset();

	public abstract void render(EnvRenderer<T, U> renderer);

	public abstract void seed(long seed);

	public abstract void close() throws Exception;
}
