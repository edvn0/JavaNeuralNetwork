package demos.rl;

import reinforcement.games.SimpleGame;
import reinforcement.learning.LearnableEnvironment;
import reinforcement.learning.dqn.DQNAgent;

public class Simple {

	public static void main(String[] args) {

		SimpleGame g = new SimpleGame();
		DQNAgent<Integer> agent = new DQNAgent<>(g, 0.01, 0.99, 15);
		LearnableEnvironment<Integer> learnableEnvironment = new LearnableEnvironment<>(g,
			agent, null) {
		};
		learnableEnvironment.fit(300);

		agent.setIsTraining(false);


	}

}
