package demos;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import utilities.NetworkUtilities;

public class MNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesTest;

	public static void main(String[] args) throws IOException {

		int epochs = Integer.parseInt(args[0]);
		int batch = Integer.parseInt(args[1]);

		ActivationFunction[] functions = new ActivationFunction[4];
		functions[0] = new ReluFunction();
		functions[1] = new ReluFunction();
		functions[2] = new ReluFunction();
		functions[3] = new SoftmaxFunction();
		ErrorFunction function = new CrossEntropyErrorFunction();
		EvaluationFunction eval = new MnistEvaluationFunction();
		NeuralNetwork network = new NeuralNetwork(0.035, functions, function, eval,
			new int[]{784, 100, 100, 10});
		System.out.println("Initialized network.");

		System.out.println("Difference: " +
			((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) * (9.357E-7))
			+ "Mb\nTotal: "
			+ (Runtime.getRuntime().totalMemory() * (9.357E-7)) + "Mb\nFree: " + (
			Runtime.getRuntime()
				.freeMemory() * (9.357E-7)) + "Mb");

		boolean existsMac = Files.exists(Paths.get(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv"));
		boolean existsVM = Files
			.exists(Paths.get("/home/edwin98carlsson/mnist-in-csv/mnist_train.csv"));

		String base = "";
		String pathTrain = null;
		String pathTest = null;
		if (existsMac && !existsVM) {
			pathTrain = "/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv";
			pathTest = "/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_test.csv";
			base = "/Users/edwincarlsson/Downloads";
		} else if (!existsMac && existsVM) {
			pathTrain = "/home/edwin98carlsson/mnist-in-csv/mnist_train.csv";
			pathTest = "/home/edwin98carlsson/mnist-in-csv/mnist_test.csv";
			base = "/home/edwin98carlsson/";
		}

		imagesTrain = generateDataFromCSV(pathTrain);
		imagesTest = generateDataFromCSV(pathTest);
		System.out.println("Difference: " +
			((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) * (9.357E-7))
			+ "Mb\nTotal: "
			+ (Runtime.getRuntime().totalMemory() * (9.357E-7)) + "Mb\nFree: " + (
			Runtime.getRuntime()
				.freeMemory() * (9.357E-7)) + "Mb");
		System.out.println("Initialized files.");

		System.out.println("Starting SGD...");
		network.stochasticGradientDescent(imagesTrain, imagesTest, epochs, batch);
		System.out.println("Finished SGD!");
		network.outputChart(base);
		network.writeObject(base);
	}

	private static List<NetworkInput> generateDataFromCSV(String path) throws IOException {
		return Files.lines(Paths.get(path)).
			map(e -> e.split(",")).
			map(NetworkUtilities::apply).
			collect(Collectors.toList());
	}
}
