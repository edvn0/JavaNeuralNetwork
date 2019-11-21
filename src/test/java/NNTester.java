import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import math.activations.ActivationFunction;
import math.activations.TanhFunction;
import math.errors.ErrorFunction;
import math.errors.MeanSquaredErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.XOREvaluationFunction;
import matrix.Matrix;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.SingleLayerPerceptron;
import neuralnetwork.Trainable;

public class NNTester {

	public static void main(String[] args) {

		ActivationFunction[] functions = new ActivationFunction[4];
		functions[0] = new TanhFunction();
		functions[1] = new TanhFunction();
		functions[2] = new TanhFunction();
		functions[3] = new TanhFunction();
		ErrorFunction function = new MeanSquaredErrorFunction();
		EvaluationFunction eval = new XOREvaluationFunction();

		SingleLayerPerceptron perceptron = new SingleLayerPerceptron(2, 5, 1, 0.001);
		NeuralNetwork network = new NeuralNetwork(0.001, functions, function, eval,
			new int[]{2, 5, 5, 1});

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

		List<Matrix[]> trainingData = new ArrayList<>();
		List<Matrix[]> testData = new ArrayList<>();
		for (int i = 0; i < 10000; i++) {
			int random = rs.nextInt(size);
			Matrix[] input = new Matrix[]{training[random], correct[random]};
			trainingData.add(input);
			testData.add(input);
		}

		Collections.shuffle(testData);

		for (Matrix[] testDatum : testData) {
			perceptron.train(testDatum[0], testDatum[1]);
		}

		network.stochasticGradientDescent(trainingData, testData, 100, 2);

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
	}

}
