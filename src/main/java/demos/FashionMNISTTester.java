package demos;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.error_functions.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import optimizers.ADAM;
import utilities.NetworkUtilities;

public class FashionMNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesValidate;

	public static void main(final String[] args) throws IOException {
		long tMem, fMem;

		NeuralNetwork network = new NeuralNetwork(new NetworkBuilder(7).setFirstLayer(784)
				.setLayer(35, new LeakyReluFunction(0.01)).setLayer(35, new LeakyReluFunction(0.01))
				.setLayer(35, new LeakyReluFunction(0.01)).setLayer(35, new LeakyReluFunction(0.01))
				.setLayer(35, new LeakyReluFunction(0.01)).setLastLayer(10, new SoftmaxFunction())
				.setCostFunction(new CrossEntropyCostFunction()).setEvaluationFunction(new ArgMaxEvaluationFunction())
				.setOptimizer(new ADAM(0.001, 0.9, 0.999)));

		network.display();
		System.out.println("Initialized network.");

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information prior to file reading:");
		System.out.printf("Total Memory: %.3fMB%n", tMem / (1024.0 * 1024.0));
		System.out.printf("Free Memory: %.3fMB", fMem / (1024.0 * 1024.0));
		System.out.println();

		final boolean existsMac = Files
				.exists(Paths.get("/Users/edwincarlsson/Downloads/fashionmnist/fashion-mnist_train.csv"));
		final boolean existsVM = Files.exists(Paths.get("/home/edwin98carlsson/fashionmnist/fashion-mnist_train.csv"));
		final boolean existsWindows;/*
									 * Files
									 * .exists(Paths.get("/home/edwin98carlsson/mnist-in-csv/mnist_train.csv"))
									 */

		String base = "";
		String pathTrain = null;
		String pathTest = null;
		if (existsMac) {
			pathTrain = "/Users/edwincarlsson/Downloads/fashionmnist/fashion-mnist_train.csv";
			pathTest = "/Users/edwincarlsson/Downloads/fashionmnist/fashion-mnist_test.csv";
			base = "/Users/edwincarlsson/Downloads";
		} else if (existsVM) {
			pathTrain = "/home/edwin98carlsson/fashionmnist/fashion-mnist_train.csv";
			pathTest = "/home/edwin98carlsson/fashionmnist/fashion-mnist_test.csv";
			base = "/home/edwin98carlsson/";
		}

		imagesTrain = generateDataFromCSV(pathTrain);
		imagesValidate = generateDataFromCSV(pathTest);

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information after reading files:");
		System.out.printf("Total Memory: %.3fMB%n", tMem / (1024.0 * 1024.0));
		System.out.printf("Free Memory: %.3fMB%n", fMem / (1024.0 * 1024.0));
		System.out.println();

		Collections.shuffle(imagesTrain);
		Collections.shuffle(imagesValidate);

		final List<NetworkInput> imagesTest = imagesTrain.subList(0, (int) (imagesTrain.size() * 0.1));

		imagesTest.addAll(imagesValidate.subList(0, (int) (imagesValidate.size() * 0.1)));

		System.out.println("Starting gradient descent...");
		network.trainWithMetrics(imagesTrain, imagesValidate, 10, 128, true, base);
		System.out.println("Finished gradient descent!");
		System.out.println();
		System.out.println("Evaluating the test data.");
		double correct = network.evaluateTestData(imagesTest, 100);
		System.out.println("Correct evaluation percentage: " + correct + ".");
		System.out.println("Writing charts. Serialising.");
		network.writeObject(base);
	}

	private static List<NetworkInput> generateDataFromCSV(final String path) throws IOException {
		try (var out = Files.lines(Paths.get(path))) {
			return out.map(e -> e.split(",")).map(NetworkUtilities::MNISTApply).collect(Collectors.toList());
		} catch (IOException e) {
			System.out.println(e);
		}
		return Collections.emptyList();
	}

}
