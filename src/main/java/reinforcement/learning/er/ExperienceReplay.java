package reinforcement.learning.er;

import static java.security.SecureRandom.getInstanceStrong;

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import reinforcement.utils.EnvObservation;

public class ExperienceReplay {

	private final int sampleSize;
	private final int maxSize;
	private final EnvObservation[] states;
	private final int[] actions;
	private final double[] rewards;
	private final EnvObservation[] newStates;
	private final int[] dones;
	private final SecureRandom random;
	private int cursor;

	public ExperienceReplay(int sampleSize, int maxSize) {
		this.sampleSize = sampleSize;
		this.maxSize = maxSize;
		this.states = new EnvObservation[maxSize];
		this.actions = new int[maxSize];
		this.rewards = new double[maxSize];
		this.newStates = new EnvObservation[maxSize];
		this.dones = new int[maxSize];

		SecureRandom tempRandom;
		try {
			tempRandom = getInstanceStrong();
		} catch (NoSuchAlgorithmException e) {
			tempRandom = new SecureRandom();
		}

		this.random = tempRandom;
		this.cursor = 0;
	}

	public void store(Transition s) {
		this.store(s.getS(), s.getAction(), s.getReward(), s.getNewS(), s.isDone());
	}

	public void store(EnvObservation s, int action, double reward, EnvObservation newS,
		boolean done) {
		this.cursor = this.cursor % this.maxSize;

		states[cursor] = s;
		actions[cursor] = action;
		rewards[cursor] = reward;
		newStates[cursor] = newS;
		dones[cursor] =
			done ? 0 : 1; // When performing bellman, we want to remove TD if we are done.

		this.cursor++;
	}

	public List<Transition> sample() {
		Set<Integer> generated = new LinkedHashSet<>();
		int toSample = Math.min(this.sampleSize, this.cursor);
		while (generated.size() < toSample) {
			Integer next = this.random.nextInt(this.cursor);
			// As we're adding to a set, this will automatically do a containment check
			generated.add(next);
		}

		return generated.stream().map(
			e -> new Transition(this.states[e],
				this.actions[e],
				this.rewards[e],
				this.newStates[e],
				this.dones[e]))
			.collect(Collectors.toList());
	}
}
