package demos.rl;

import reinforcement.games.SimpleGame;
import reinforcement.learning.LearnableEnvironment;
import reinforcement.learning.dqn.DQNAgent;

public class Simple {

	public static void main(String[] args) {

		SimpleGame g = new SimpleGame();
		DQNAgent<Integer> agent = new DQNAgent<>(g, 0.1, 0.95, 15);
		var learnableEnvironment = LearnableEnvironment.of(g, agent, null);
		learnableEnvironment.fit(1000);

		agent.setIsTraining(false);

		var sars = g.reset();
		boolean done = sars.isDone();
		var obs = sars.getObservation();
		while (!done) {
			g.render();
			var newObs = g.step(agent.act(obs));
			obs = newObs.getObservation();
			done = newObs.isDone();
		}
	}

}
