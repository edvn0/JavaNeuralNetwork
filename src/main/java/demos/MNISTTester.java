package demos;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import math.activations.SigmoidFunction;
import math.activations.SoftmaxFunction;
import math.activations.TanhFunction;
import math.errors.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.NeuralNetwork.NetworkBuilder;
import utilities.NetworkUtilities;

public class MNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesValidate;

	public static void main(final String[] args) throws IOException {

		final int epochs = Integer.parseInt(args[0]);
		final int batch = Integer.parseInt(args[1]);
		final double learningRate = Double.parseDouble(args[2]);

		NeuralNetwork network = new NeuralNetwork(
			new NetworkBuilder(4)
				.setFirstLayer(784)
				.setLayer(100, new SigmoidFunction())
				.setLayer(100, new SigmoidFunction())
				.setLastLayer(10, new SoftmaxFunction())
				.setCostFunction(new CrossEntropyCostFunction())
				.setEvaluationFunction(new ArgMaxEvaluationFunction())
				.setLearningRate(learningRate)
		);
		System.out.println("Initialized network.");

		System.out.println(
			"Difference: " + (
				(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
					* (9.357E-7))
				+ "Mb\nTotal: " + (Runtime.getRuntime().totalMemory() * (9.357E-7)) + "Mb\nFree: "
				+ (Runtime.getRuntime().freeMemory() * (9.357E-7)) + "Mb");

		final boolean existsMac = Files
			.exists(Paths.get("/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv"));
		final boolean existsVM = Files
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
		imagesValidate = generateDataFromCSV(pathTest);
		System.out.println(
			"Difference: " + (
				(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
					* (9.357E-7))
				+ "Mb\nTotal: " + (Runtime.getRuntime().totalMemory() * (9.357E-7)) + "Mb\nFree: "
				+ (Runtime.getRuntime().freeMemory() * (9.357E-7)) + "Mb");
		System.out.println("Initialized files.");

		Collections.shuffle(imagesTrain);
		Collections.shuffle(imagesValidate);

		final List<NetworkInput> imagesTest = imagesTrain
			.subList(0, (int) (imagesTrain.size() * 0.1));

		imagesTest.addAll(imagesValidate.subList(0, (int) (imagesValidate.size() * 0.1)));

		System.out.println("Starting SGD...");
		network.stochasticGradientDescent(imagesTrain, imagesValidate, epochs, batch);
		System.out.println("Finished SGD!");
		System.out.println("Evaluating the test data.");
		double correct = network.evaluateTestData(imagesTest, 100);
		System.out.println("Correct evaluation percentage: " + correct + ".");
		System.out.println("Writing charts and serialisation.");
		network.outputChart(base);
		network.writeObject(base);
	}

	private static List<NetworkInput> generateDataFromCSV(final String path) throws IOException {
		return Files.lines(Paths.get(path)).map(e -> e.split(",")).map(NetworkUtilities::apply)
			.collect(Collectors.toList());
	}
}
