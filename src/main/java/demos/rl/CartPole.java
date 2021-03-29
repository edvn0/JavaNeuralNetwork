package demos.rl;

import reinforcement.learning.LearnableEnvironment;
import reinforcement.learning.dqn.DQNAgent;

public class CartPole {

	public static void main(String[] args) {

		reinforcement.games.CartPole game = new reinforcement.games.CartPole();
		DQNAgent<Double> agent = new DQNAgent<>(game, 1, 0.95, 70);

		var learnable = LearnableEnvironment.of(game, agent, null);

		learnable.fit(35000);

		var initial = game.reset();
		while (!initial.isDone()) {
			var act = agent.act(initial.getObservation());
			initial = game.step(act);
			System.out.println(initial.getReward());
		}

	}

}
