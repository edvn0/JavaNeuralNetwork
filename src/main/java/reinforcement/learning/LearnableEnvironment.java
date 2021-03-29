package reinforcement.learning;

import java.util.ArrayList;
import java.util.List;
import reinforcement.env.BaseEnvironment;
import reinforcement.learning.agent.LearningAgent;
import reinforcement.learning.er.ExperienceReplay;

public abstract class LearnableEnvironment<ObsT> {

	private static final int INFO_INTERVAL = 50;
	private final BaseEnvironment<Integer, ObsT> env;
	private final LearningAgent<ObsT> agent;

	// These are trackers for the agent
	private final List<Double> rewards;
	private final List<Boolean> won;
	private final ExperienceReplay replay;

	public LearnableEnvironment(BaseEnvironment<Integer, ObsT> env,
		LearningAgent<ObsT> agent, final ExperienceReplay replay) {
		this.env = env;
		this.agent = agent;
		this.replay = replay == null ? new ExperienceReplay(64, 100_000) : replay;

		this.rewards = new ArrayList<>();
		this.won = new ArrayList<>();
	}

	public static <T, U> LearnableEnvironment<U> of(BaseEnvironment<Integer, U> env,
		LearningAgent<U> agent,
		final ExperienceReplay replay) {
		return new LearnableEnvironment<U>(env, agent, replay) {
		};
	}

	/**
	 * Fits the agent to the environment and outputs plots and serialized model.
	 *
	 * @param epochs          how many epochs
	 * @param outputDirectory which directory?
	 */
	public void fitAndSave(int epochs, String outputDirectory) {
		learnOnEnvironment(epochs);
		this.agent.serialize(outputDirectory);
		plotLearning(outputDirectory);
	}

	private void learnOnEnvironment(final int epochs) {
		boolean isAgentTraining = agent.isTraining();
		for (int i = 0; i < epochs; i++) {
			var state = this.env.reset();
			double averageReward = 0d;
			while (!state.isDone()) {
				int action = this.agent.act(state.getObservation());
				var newState = this.env.step(action);
				if (isAgentTraining) {
					this.replay.store(state.getObservation(), action, newState.getReward(),
						newState.getObservation(), newState.isDone());
					this.agent.learn(this.replay.sample());
				}
				state = newState;
				averageReward += newState.getReward();
			}

			if (i != 0 && i % INFO_INTERVAL == 0) {
				var mean = rewards
					.subList(Math.max(0, rewards.size() - INFO_INTERVAL), rewards.size()).stream()
					.mapToDouble(e -> e).average().orElseThrow();

				var fGoldAvg = won
					.subList(Math.max(0, won.size() - INFO_INTERVAL), won.size()).stream()
					.mapToDouble(e -> e ? 1d : 0d).average().orElseThrow();

				System.out.printf(
					"Episode: %d, Average Score: %f, Won Percentage: %f\n", i,
					mean, fGoldAvg * 100);
			}

			this.rewards.add(averageReward);
			this.won.add(this.env.didWin());
		}
	}

	private void plotLearning(final String outputPath) {
		// File rewards = new File(outputPath);
		// File wonPercentage = new File(outputPath);
	}

	public void fit(int epochs) {
		learnOnEnvironment(epochs);
	}

}
