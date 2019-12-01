package neuralnetwork;

import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import math.activations.ActivationFunction;
import math.activations.TanhFunction;
import math.errors.ErrorFunction;
import math.errors.MeanSquaredErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.XOREvaluationFunction;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class NNTester {

	public static void main(String[] args) throws IOException {

		ActivationFunction[] functions = new ActivationFunction[3];
		functions[0] = new TanhFunction();
		functions[1] = new TanhFunction();
		functions[2] = new TanhFunction();
		ErrorFunction function = new MeanSquaredErrorFunction();
		EvaluationFunction eval = new XOREvaluationFunction();

		SingleLayerPerceptron perceptron = new SingleLayerPerceptron(2, 5, 1, 0.01);
		NeuralNetwork network = new NeuralNetwork(0.000001, functions, function, eval,
			new int[]{2, 6, 1});

		double score = network.getScore();
		System.out.println(score);

		Trainable[] trainable = new Trainable[2];
		trainable[0] = network;
		trainable[1] = perceptron;
		Matrix[] training = {
			Matrix.Factory.importFromArray(new double[][]{{1d}, {1d}}),
			Matrix.Factory.importFromArray(new double[][]{{0d}, {0d}}),
			Matrix.Factory.importFromArray(new double[][]{{1d}, {0d}}),
			Matrix.Factory.importFromArray(new double[][]{{0d}, {1d}}),
		};

		Matrix[] correct = {
			Matrix.Factory.importFromArray(new double[][]{{0d}}),
			Matrix.Factory.importFromArray(new double[][]{{0d}}),
			Matrix.Factory.importFromArray(new double[][]{{1d}}),
			Matrix.Factory.importFromArray(new double[][]{{1d}}),
		};

		int size = correct.length;
		SecureRandom rs = new SecureRandom();

		List<NetworkInput> trainingData = new ArrayList<>();
		List<NetworkInput> testData = new ArrayList<>();
		for (int i = 0; i < 10000; i++) {
			int random = rs.nextInt(size);
			NetworkInput input = new NetworkInput((DenseMatrix) training[random],
				(DenseMatrix) correct[random]);
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
						"Predict[" + i + "] : " + t.predict((DenseMatrix) training[i]));
				}
			}
			System.out.println();
		}

		double newScore = network.getScore();
		if (newScore > score) {
			network.writeObject(
				"/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/test/resources");
		}
	}

}
