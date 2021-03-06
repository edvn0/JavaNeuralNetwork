package reinforcement.env.renderer;

import reinforcement.env.BaseEnvironment;

public abstract class EnvRenderer<ActionT, ObsT> {

	protected BaseEnvironment<ActionT, ObsT> env;

	public EnvRenderer(BaseEnvironment<ActionT, ObsT> env) {
		this.env = env;
	}

	public abstract void render();

	protected void clearScreen() {
		System.out.print("\033[H\033[2J");
		System.out.flush();
	}
}
