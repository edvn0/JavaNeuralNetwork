package geneticalgorithm;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.DoubleStream;

public class Agent implements Measurable, Comparable {

	private double[] genes;
	private double fitness;

	private double target;

	private static final SecureRandom random = new SecureRandom();


	public Agent(double[] genes, double fitness, double target) {
		this.genes = genes;
		this.fitness = fitness;
		this.target = target;
	}

	public Agent(int size, double target) {
		this.genes = new double[size];
		for (int i = 0; i < size; i++) {
			this.genes[i] = (random.nextDouble() * 2) - 1;
		}
		this.target = target;
		this.fitness = 0;
	}

	/**
	 * Measure fitness by sum of the genes divided by the target value.
	 */
	@Override
	public void measureFitness() {
		double sum = DoubleStream.of(genes).sum() / this.target;
		this.fitness = sum;
	}

	public Agent crossOver(Agent with) {
		Agent child = new Agent(this.genes.length, this.target);

		int size = this.genes.length;

		int mid = random.nextInt(size);

		for (int i = 0; i < size; i++) {
			double tGene = this.genes[i];
			double wGene = with.genes[i];
			if (i > mid) {
				child.setGene(i, tGene);
			} else {
				child.setGene(i, wGene);
			}
		}
		return child;
	}

	public void mutate(double rate) {
		for (int i = 0; i < this.genes.length; i++) {
			if (random.nextDouble() < rate) {
				this.genes[i] = (random.nextDouble() * 2) - 1;
			}
		}
	}

	public double getFitness() {
		return this.fitness;
	}

	public void setGene(int i, double gene) {
		this.genes[i] = gene;
	}

	public double getTarget() {
		return target;
	}

	public void setTarget(double target) {
		this.target = target;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof Agent)) {
			return false;
		}
		Agent agent = (Agent) o;
		return Double.compare(agent.fitness, fitness) == 0 &&
			Arrays.equals(genes, agent.genes);
	}

	@Override
	public int hashCode() {
		int result = Objects.hash(fitness);
		result = 31 * result + Arrays.hashCode(genes);
		return result;
	}

	@Override
	public int compareTo(Object o) {
		if (this.equals(o)) {
			return 0;
		}
		if (!(o instanceof Agent)) {
			return 0;
		}
		Agent agent = (Agent) o;
		return this.getFitness() - agent.getFitness() > 0 ? 1 : -1;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("Agent{");
		sb.append("Fitness=").append(fitness);
		sb.append(", Genes=").append(Arrays.toString(genes));
		sb.append('}');
		return sb.toString();
	}
}
