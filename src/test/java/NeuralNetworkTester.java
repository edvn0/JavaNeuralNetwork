import neuralnetwork.NeuralNetwork;

public class NeuralNetworkTester {


	public static void main(String[] args) {
		NeuralNetwork network = new NeuralNetwork(2, 2, 7, 1, 0.1);

		network.train(new double[]{0d, 0d}, new double[]{0});
		network.train(new double[]{1d, 0d}, new double[]{1});
		network.train(new double[]{0d, 1d}, new double[]{1});
		network.train(new double[]{1d, 1d}, new double[]{0});
	}

}
