import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import matrix.Matrix;
import neuralnetwork.SingleLayerPerceptron;
import neuralnetwork.Trainable;

public class NNTester {

	public static void main(String[] args) {

		SingleLayerPerceptron perceptron = new SingleLayerPerceptron(2, 5, 1, 0.1);

		Trainable[] trainable = new Trainable[2];
		//trainable[0] = network;
		trainable[1] = perceptron;
		Matrix[] training = {
			new Matrix(new double[][]{{1d}, {1d}}),
			new Matrix(new double[][]{{0d}, {0d}}),
			new Matrix(new double[][]{{1d}, {0d}}),
			new Matrix(new double[][]{{0d}, {1d}}),
		};

		Matrix[] correct = {
			new Matrix(new double[][]{{0d}}),
			new Matrix(new double[][]{{0d}}),
			new Matrix(new double[][]{{1d}}),
			new Matrix(new double[][]{{1d}}),
		};

		int size = correct.length;
		SecureRandom rs = new SecureRandom();

		List<Matrix[]> trainingData = new ArrayList<>();
		for (int i = 0; i < 10000; i++) {
			int random = rs.nextInt(size);
			Matrix[] input = new Matrix[]{training[random],correct[random]};
			trainingData.add(input);
		}

		for (Trainable t : trainable) {
			if (t != null) {
				t.train(trainingData, "SGD");
			}
		}

		System.out.println();
		System.out.println();
		for (Trainable t : trainable) {
			if (t != null) {
				System.out.println(t.getClass());
				for (int i = 0; i < training.length; i++) {
					System.out.println(
						"Predict[" + i + "] : " + Arrays
							.deepToString(t.predict(training[i]).getData()));
				}
			}
			System.out.println();
		}
	}

}
