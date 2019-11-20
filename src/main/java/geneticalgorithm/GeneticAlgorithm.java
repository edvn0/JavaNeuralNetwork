package geneticalgorithm;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;

public class GeneticAlgorithm {

	private Agent[] population;

	private int populationSize;

	private static final SecureRandom random = new SecureRandom();
	private double mutationRate;

	public GeneticAlgorithm(Agent[] population, double mRate) {
		this.population = population;
		this.populationSize = this.population.length;
		this.mutationRate = mRate;
	}

	public GeneticAlgorithm(int populationSize, double mRate) {
		mutationRate = mRate;
		population = new Agent[populationSize];
		for (int i = 0; i < population.length; i++) {
			population[i] = new Agent(10, 10);
		}
		this.populationSize = populationSize;
	}

	public GeneticAlgorithm(int populationSize) {
		mutationRate = 0.1;
		population = new Agent[populationSize];
		for (int i = 0; i < population.length; i++) {
			population[i] = new Agent(10, 13);
		}
		this.populationSize = populationSize;
	}

	public Agent getKFittest(int k) {
		Arrays.sort(this.population, null);
		return this.population[k - 1];
	}

	public Agent[] createMatingPool() {
		ArrayList<Agent> pool = new ArrayList<>();
		for (Agent current : population) {
			int poolSize = (int) (Math.abs(current.getFitness()) * 100);
			for (int j = 0; j < poolSize; j++) {
				pool.add(current);
			}
		}
		Agent[] poolArray = new Agent[pool.size()];
		int k = 0;
		for (Agent a : pool) {
			poolArray[k++] = a;
		}
		return poolArray;
	}

	public void poolCrossOver(Agent[] pool) {
		for (int i = 0; i < this.populationSize; i++) {
			Agent a = pool[random.nextInt(pool.length)];
			Agent b = pool[random.nextInt(pool.length)];
			Agent c = a.crossOver(b);
			c.mutate(this.mutationRate);
			this.population[i] = c;
		}
	}


	public void calculateFitness() {
		for (Agent agent : population) {
			agent.measureFitness();
		}
	}

	public Agent[] getAgents() {
		return this.population;
	}
}
