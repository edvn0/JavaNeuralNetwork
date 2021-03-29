package demos.rl;

import reinforcement.learning.LearnableEnvironment;
import reinforcement.learning.dqn.DQNAgent;

public class CartPole {

	public static void main(String[] args) {

		reinforcement.games.CartPole game = new reinforcement.games.CartPole();
		DQNAgent<Double> agent = new DQNAgent<>(game, 0.00115, 0.95, 200);

		var learnable = LearnableEnvironment.of(game, agent, null);

		learnable.fit(1000);

	}

}
