package utilites;

public class NeuralNetworkOptions {

	private double learningRate;
	private double iterations;

	public NeuralNetworkOptions() {
	}

	private NeuralNetworkOptions(double learningRate, double iterations) {
		this.learningRate = learningRate;
		this.iterations = iterations;
	}

	public NeuralNetworkOptions create() {
		return new NeuralNetworkOptions(learningRate, iterations);
	}

	public static NeuralNetworkOptions getInstance() {
		return new NeuralNetworkOptions();
	}

	/**
	 * Mutators
	 */

	/**
	 * sets the learning rate of this neural network
	 *
	 * @param v learning rate, typically from 10e-3 to 10e-5
	 */
	public NeuralNetworkOptions setLearningRate(double v) {
		this.learningRate = v;
		return this;
	}

	public NeuralNetworkOptions setTrainingIterations(int iterations) {
		this.iterations = iterations;
		return this;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public double getIterations() {
		return iterations;
	}

	public String toString() {
		return "Iterations:" + this.iterations + "\nLearning Rate: " + this.learningRate;
	}
}