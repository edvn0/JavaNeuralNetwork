package neuralnetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import math.activations.ActivationFunction;
import math.activations.SigmoidFunction;
import math.activations.SoftmaxFunction;
import math.activations.TanhFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;
import matrix.Matrix;

public class MNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesTest;

	public static void main(String[] args) throws IOException {
		ActivationFunction[] functions = new ActivationFunction[5];
		functions[0] = new SigmoidFunction();
		functions[1] = new SigmoidFunction();
		functions[2] = new SigmoidFunction();
		functions[3] = new TanhFunction();
		functions[4] = new SoftmaxFunction();
		ErrorFunction function = new CrossEntropyErrorFunction();
		EvaluationFunction eval = new MnistEvaluationFunction();
		NeuralNetwork network = new NeuralNetwork(5e-3, functions, function, eval,
			new int[]{784, 30, 30, 30, 10});

		System.out.println("Starting bGD");
		batchGradientDescentKindOf(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv", 10000 / 10, network);
		System.out.println("Ending sGD.");

		/*imagesTrain = generateDataFromCSV(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv");
		imagesTest = generateDataFromCSV(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_test.csv");

		System.out.println("Starting SGD...");
		network.stochasticGradientDescent(imagesTrain, imagesTest, 30, 32);
		System.out.println("Finished SGD!");*/
	}

	private static void batchGradientDescentKindOf(String s, int i, NeuralNetwork neuralNetwork) {
		List<NetworkInput> temp = new ArrayList<>();

		int k = 0;
		try (Scanner n = new Scanner(new File(s))) {
			while (n.hasNextLine()) {
				String[] line = n.nextLine().split(",");
				temp.add(getTrainData(new Matrix(normalizeData(line))));
				k++;

				if (k % i == 0) {
					neuralNetwork.stochasticGradientDescent(temp, temp, 10, 32);
					temp.clear();
				}
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	private static List<NetworkInput> generateDataFromCSV(String path) throws IOException {
		List<NetworkInput> things = new ArrayList<>();

		Files.readAllLines(Paths.get(path))
			.forEach((e) -> things.add(getTrainData(new Matrix(normalizeData(e.split(","))))));

		return things;
	}

	private static double[][] normalizeData(String[] split) {
		double[][] d = new double[1 + 28 * 28][1];
		for (int i = 1; i < split.length; i++) {
			d[i][0] = Double.parseDouble(split[i]) / 255d;
		}
		d[0][0] = Integer.parseInt(split[0]);
		return d;
	}

	private static NetworkInput getTrainData(Matrix m) {
		double[][] corr = new double[10][1];
		String num = m.getSingleValue() + "";
		String newNum = num.substring(0, 1);
		switch (Integer.parseInt(newNum)) {
			case 0:
				corr[0][0] = 1;
				break;
			case 1:
				corr[1][0] = 1;
				break;
			case 2:
				corr[2][0] = 1;
				break;
			case 3:
				corr[3][0] = 1;
				break;
			case 4:
				corr[4][0] = 1;
				break;
			case 5:
				corr[5][0] = 1;
				break;
			case 6:
				corr[6][0] = 1;
				break;
			case 7:
				corr[7][0] = 1;
				break;
			case 8:
				corr[8][0] = 1;
				break;
			case 9:
				corr[9][0] = 1;
				break;
			default:
				break;
		}

		double[][] data = new double[28 * 28][1];

		int dataSize = data.length;
		for (int j = 1; j < dataSize; j++) {
			data[j - 1][0] = m.getData()[j][0];
		}
		return new NetworkInput(new Matrix(data), new Matrix(corr));
	}

}
