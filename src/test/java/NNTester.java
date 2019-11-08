import java.security.SecureRandom;
import math.TanhFunction;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.SingleLayerPerceptron;
import neuralnetwork.Trainable;

public class NNTester {

	public static void main(String[] args) {
		NeuralNetwork network = new NeuralNetwork(2, 10, 7, 1, 0.000001, new TanhFunction());
		SingleLayerPerceptron perceptron = new SingleLayerPerceptron(2, 3, 1, 0.0001);
		perceptron.setDefaultValues();

		Trainable[] trainable = new Trainable[2];
		trainable[0] = network;
		trainable[1] = perceptron;
		SecureRandom random = new SecureRandom();
		double[][] xor = {
			{0, 1},
			{1, 0},
			{0, 0},
			{1, 1},
		};

		double[][] correct = {
			{1},
			{1},
			{0},
			{0},
		};

		for (Trainable t : trainable) {
			if (t != null) {
				for (int i = 0; i < 10000; i++) {
					t.train(xor[i % 4], correct[i % 4]);
				}
			}
		}

		System.out.println();
		System.out.println();
		for (Trainable t : trainable) {
			if (t != null) {
				System.out.println(t.getClass());
				t.predict(xor[0]).show();
				t.predict(xor[1]).show();
				t.predict(xor[2]).show();
				t.predict(xor[3]).show();
			}
			System.out.println();
		}
	}

}
