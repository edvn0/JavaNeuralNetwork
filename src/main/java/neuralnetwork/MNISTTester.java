package neuralnetwork;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;
import utilities.NetworkUtilities;

public class MNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesTest;

	public static void main(String[] args) throws IOException {

		System.out.println("Initialized network.");
		ActivationFunction[] functions = new ActivationFunction[6];
		functions[0] = new ReluFunction();
		functions[1] = new ReluFunction();
		functions[2] = new ReluFunction();
		functions[3] = new ReluFunction();
		functions[4] = new ReluFunction();
		functions[5] = new SoftmaxFunction();
		ErrorFunction function = new CrossEntropyErrorFunction();
		EvaluationFunction eval = new MnistEvaluationFunction();
		NeuralNetwork network = new NeuralNetwork(0.0015, functions, function, eval,
			new int[]{784, 2000, 1500, 1000, 500, 10});


		/*System.out.println("Starting bGD");
		batchGradientDescentKindOf(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv", 10000, network);
		System.out.println("Ending sGD.");*/

		boolean existsMac = Files.exists(Paths.get(
			"/Users/edwincarlsson/Programmering/Java/NeuralNetwork/mnist-in-csv/mnist_train.csv"));
		boolean existsVM = Files
			.exists(Paths.get("/home/edwin98carlsson/mnist-in-csv/mnist_train.csv"));

		String path = null;
		if (existsMac && !existsVM) {
			path = "/Users/edwincarlsson/Programmering/Java/NeuralNetwork/mnist-in-csv/mnist_train.csv";
		} else if (!existsMac && existsVM) {
			path = "/home/edwin98carlsson/mnist-in-csv/mnist_train.csv";
		}

		imagesTrain = generateDataFromCSV(path);
		imagesTest = generateDataFromCSV(path);

		System.out.println("Starting SGD...");
		network.stochasticGradientDescent(imagesTrain, imagesTest, 50, 64);
		System.out.println("Finished SGD!");
		network.outputChart("/home/edwin98carlsson/");
		network.writeObject("/home/edwin98carlsson/");
	}

	private static List<NetworkInput> generateDataFromCSV(String path) throws IOException {
		List<NetworkInput> things = new ArrayList<>();

		Files.readAllLines(Paths.get(path))
			.forEach((e) -> things.add(NetworkUtilities
				.constructDataFromDoubleArray(NetworkUtilities.normalizeData(e.split(",")))));

		return things;
	}
}
