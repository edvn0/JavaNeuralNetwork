package demos;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.NeuralNetwork.NetworkBuilder;
import optimizers.ADAM;
import utilities.NetworkUtilities;

public class FashionMNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesValidate;

	public static void main(final String[] args) throws IOException {
		long tMem, fMem;

		NeuralNetwork network = new NeuralNetwork(
			new NetworkBuilder(4)
				.setFirstLayer(784)
				.setLayer(35, new LeakyReluFunction(0.01))
				.setLayer(35, new LeakyReluFunction(0.01))
				.setLastLayer(10, new SoftmaxFunction())
				.setCostFunction(new CrossEntropyCostFunction())
				.setEvaluationFunction(new ArgMaxEvaluationFunction())
				.setOptimizer(new ADAM(0.001, 0.9, 0.999))
		);
		System.out.println("Initialized network.");

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information prior to file reading:");
		System.out
			.printf("Total Memory: %.3fMB%n", tMem / (1024.0 * 1024.0));
		System.out
			.printf("Free Memory: %.3fMB", fMem / (1024.0 * 1024.0));
		System.out.println();

		final boolean existsMac = Files
			.exists(
				Paths.get("/Users/edwincarlsson/Downloads/fashionmnist/fashion-mnist_train.csv"));
		final boolean existsVM = Files
			.exists(Paths.get("/home/edwin98carlsson/fashionmnist/fashion-mnist_train.csv"));
		final boolean existsWindows;/*Files
			.exists(Paths.get("/home/edwin98carlsson/mnist-in-csv/mnist_train.csv"))*/

		String base = "";
		String pathTrain = null;
		String pathTest = null;
		if (existsMac && !existsVM) {
			pathTrain = "/Users/edwincarlsson/Downloads/fashionmnist/fashion-mnist_train.csv";
			pathTest = "/Users/edwincarlsson/Downloads/fashionmnist/fashion-mnist_test.csv";
			base = "/Users/edwincarlsson/Downloads";
		}

		imagesTrain = generateDataFromCSV(pathTrain);
		imagesValidate = generateDataFromCSV(pathTest);

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information after reading files:");
		System.out
			.printf("Total Memory: %.3fMB\n", tMem / (1024.0 * 1024.0));
		System.out
			.printf("Free Memory: %.3fMB\n", fMem / (1024.0 * 1024.0));
		System.out.println();

		Collections.shuffle(imagesTrain);
		Collections.shuffle(imagesValidate);

		final List<NetworkInput> imagesTest = imagesTrain
			.subList(0, (int) (imagesTrain.size() * 0.1));

		imagesTest.addAll(imagesValidate.subList(0, (int) (imagesValidate.size() * 0.1)));

		System.out.println("Starting gradient descent...");
		network.trainWithMetrics(imagesTrain, imagesValidate, 64, 64, true, base);
		System.out.println("Finished gradient descent!");
		System.out.println();
		System.out.println("Evaluating the test data.");
		double correct = network.evaluateTestData(imagesTest, 100);
		System.out.println("Correct evaluation percentage: " + correct + ".");
		System.out.println("Writing charts. Serialising.");
		network.writeObject(base);
	}

	private static List<NetworkInput> generateDataFromCSV(final String path) throws IOException {
		return Files
			.lines(Paths.get(path))
			.map(e -> e.split(","))
			.map(NetworkUtilities::MNISTApply)
			.collect(Collectors.toList());
	}


}
