package demos.rl;

import reinforcement.games.WumpusGame;
import reinforcement.learning.LearnableEnvironment;
import reinforcement.learning.dqn.DQNAgent;

public class Wumpus {

	public static void main(String[] args) throws InterruptedException {
		WumpusGame game = new WumpusGame(4);
		DQNAgent<Integer> agent = new DQNAgent<>(game, 0.01, 0.99, 15);
		LearnableEnvironment<Integer> learnableEnvironment = new LearnableEnvironment<>(game,
			agent, null) {
		};
		learnableEnvironment.fit(1000);

	}

}
