package geneticalgorithm;

public class GeneticAlgorithmDemo {

	public static GeneticAlgorithm population;

	public GeneticAlgorithmDemo() {
		population = new GeneticAlgorithm(30, 0.1);
	}

	public static void main(String[] args) {
		new GeneticAlgorithmDemo();
		for (int i = 0; i < 10000; i++) {
			population.calculateFitness();
			Agent[] pool = population.createMatingPool();
			population.poolCrossOver(pool);
		}

		for (Agent p : population.getAgents()) {
			System.out.println(p);

		}
	}


}
