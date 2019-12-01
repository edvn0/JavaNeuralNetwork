package neuralnetwork;

import java.io.IOException;

public class NNTester {

	public static void main(String[] args) throws IOException {

		/*ActivationFunction[] functions = new ActivationFunction[10];
		functions[0] = new TanhFunction();
		functions[1] = new TanhFunction();
		functions[2] = new TanhFunction();
		functions[3] = new TanhFunction();
		functions[4] = new TanhFunction();
		functions[5] = new TanhFunction();
		functions[6] = new TanhFunction();
		functions[7] = new TanhFunction();
		functions[8] = new TanhFunction();
		functions[9] = new TanhFunction();
		ErrorFunction function = new MeanSquaredErrorFunction();
		EvaluationFunction eval = new XOREvaluationFunction();

		SingleLayerPerceptron perceptron = new SingleLayerPerceptron(2, 5, 1, 0.01);
		NeuralNetwork network = new NeuralNetwork(1, functions, function, eval,
			new int[]{2, 3, 3, 3, 3, 3, 3, 3, 3, 1});

		double score = network.getScore();
		System.out.println(score);

		Trainable[] trainable = new Trainable[2];
		trainable[0] = network;
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

		List<NetworkInput> trainingData = new ArrayList<>();
		List<NetworkInput> testData = new ArrayList<>();
		for (int i = 0; i < 10000; i++) {
			int random = rs.nextInt(size);
			NetworkInput input = new NetworkInput(training[random], correct[random]);
			trainingData.add(input);
			testData.add(input);
		}

		Collections.shuffle(testData);

		for (int i = 0; i < 100; i++) {
			for (NetworkInput testDatum : testData) {
				perceptron.train(testDatum.getData(), testDatum.getLabel());
			}
		}

		network.stochasticGradientDescent(trainingData, testData, 100, 32);

		System.out.println();
		System.out.println();
		for (
			Trainable t : trainable) {
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

		double newScore = network.getScore();
		if (newScore > score) {
			network.writeObject(
				"/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/test/resources");
		}*/
	}

}
