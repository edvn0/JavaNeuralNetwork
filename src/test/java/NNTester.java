import java.security.SecureRandom;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.SingleLayerPerceptron;
import neuralnetwork.Trainable;

public class NNTester {

	public static void main(String[] args) {
		NeuralNetwork network = new NeuralNetwork(2, 4, 12, 2, 0.1);
		SingleLayerPerceptron perceptron = new SingleLayerPerceptron(2, 3, 2, 0.1);
		perceptron.setDefaultValues();

		Trainable[] trainable = new Trainable[2];
		trainable[0] = network;
		trainable[1] = perceptron;
		SecureRandom random = new SecureRandom();
		double[][] xor = {
			{0, 0},
			{0, 1},
			{1, 1},
			{1, 0},
		};

		double[][] correct = {
			{1, 0},
			{0, 1},
			{0, 1},
			{1, 0},
		};

		for (int i = 0; i < 1; i++) {
			int r = random.nextInt(4);
			for (Trainable t : trainable) {
				if (t != null) {
					t.train(xor[r], correct[r]);
				}
			}
		}

		for (Trainable t : trainable) {
			if (t != null) {
				t.predict(xor[0]).show();
				t.predict(xor[1]).show();
				t.predict(xor[2]).show();
				t.predict(xor[3]).show();
			}
		}
	}

}
