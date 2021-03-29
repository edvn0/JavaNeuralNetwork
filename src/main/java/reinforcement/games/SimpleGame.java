package reinforcement.games;

import java.security.SecureRandom;
import java.util.function.Function;
import lombok.Getter;
import reinforcement.env.BaseEnvironment;
import reinforcement.env.renderer.EnvRenderer;
import reinforcement.env.space.Discrete;
import reinforcement.utils.EnvInfo;
import reinforcement.utils.EnvObservation;
import reinforcement.utils.Sars;

public class SimpleGame extends BaseEnvironment<Integer, Integer> {

	private static final int GAME_SIZE = 50;
	private final Function<Integer, Double> reward = e -> 1.0f - (Math.abs(e - GAME_SIZE)
		/ (double) GAME_SIZE);
	private final Function<Integer, Boolean> gameOver = e -> e == GAME_SIZE || e < 0;
	@Getter
	private int position;

	public SimpleGame() {
		this.actionSpace = new Discrete(2);
		this.observationSpace = new Discrete(GAME_SIZE);
		this.position = 0;

		this.renderer = new EnvRenderer<>(this) {
			@Override
			public void render() {
				clearScreen();
				StringBuilder b = new StringBuilder();
				for (int i = 0; i < GAME_SIZE; i++) {
					if (position == i) {
						b.append('P');
					} else {
						b.append('*');
					}
				}
				b.append("\n");
				System.out.println(b.toString());
			}
		};

	}

	@Override
	public Sars step(final Integer a) {
		if (!this.actionSpace.contains(a)) {
			EnvInfo info = new EnvInfo();
			info.addInfo("Invalid action.");
			return new Sars(this.observe(), 0, true, info);
		}

		if (a == 1) {
			this.position += 1;
		} else {
			this.position -= 1;
		}

		var s = this.observe();
		var r = this.reward.apply(this.position);

		return new Sars(s, r, this.gameOver.apply(this.position), null);
	}

	@Override
	public Sars reset() {
		this.position = 0;
		return new Sars(this.observe(), 0, false, null);
	}

	@Override
	public void render() {
		this.renderer.render();
	}

	@Override
	public void seed(final long seed) {
		if (this.random == null) {
			this.random = new SecureRandom();
		}
		this.random.setSeed(seed);
	}

	public boolean didWin() {
		return this.position >= GAME_SIZE;
	}

	@Override
	public void close() throws Exception {

	}

	@Override
	protected EnvObservation observe() {
		double[] obs = new double[GAME_SIZE];
		if (this.position < 0) {
			obs[0] = 1;
		} else if (this.position >= 50) {
			obs[obs.length - 1] = 1;
		} else {
			obs[this.position] = 1;
		}
		return new EnvObservation(obs);
	}

	public boolean isGameOver() {
		return this.gameOver.apply(this.position);
	}
}
