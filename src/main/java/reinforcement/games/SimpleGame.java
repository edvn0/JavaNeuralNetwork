package reinforcement.games;

import java.security.SecureRandom;
import java.util.function.Function;
import lombok.Getter;
import reinforcement.env.BaseEnvironment;
import reinforcement.env.renderer.EnvRenderer;
import reinforcement.env.space.Discrete;
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
		this.observationSpace = new Discrete(1);
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
				clearScreen();
				System.out.println(b.toString());
			}
		};

	}

	@Override
	public Sars step(final Integer a) {
		if (!this.actionSpace.contains(a)) {
			throw new IllegalArgumentException("Specified action does not exist in this game.");
		}

		if (a == 1) {
			this.position += 1;
		} else {
			this.position -= 1;
		}

		var s = new EnvObservation(this.position);
		var r = this.reward.apply(this.position);

		return new Sars(s, r, this.gameOver.apply(this.position), null);
	}

	@Override
	public Sars reset() {
		this.position = 0;
		return new Sars(new EnvObservation(this.position), 0, false, null);
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

	@Override
	public void close() throws Exception {

	}

	public boolean isGameOver() {
		return this.gameOver.apply(this.position);
	}

	public boolean didWin() {
		return this.position >= GAME_SIZE;
	}
}
