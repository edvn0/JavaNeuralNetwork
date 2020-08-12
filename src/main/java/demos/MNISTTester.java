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

public class MNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesValidate;

	public static void main(final String[] args) throws IOException {
		long tMem, fMem;

		if (args.length == 0)
			throw new RuntimeException("Need to supply epochs, batches.");

		final int epochs = Integer.parseInt(args[0]);
		final int batch = Integer.parseInt(args[1]);

		NeuralNetwork network = NeuralNetwork.of(new NetworkBuilder(4).setFirstLayer(784)
				.setLayer(35, new LeakyReluFunction(0.01)).setLayer(35, new LeakyReluFunction(0.01))
				.setLastLayer(10, new SoftmaxFunction()).setCostFunction(new CrossEntropyCostFunction())
				.setEvaluationFunction(new ArgMaxEvaluationFunction()).setOptimizer(new ADAM(0.001, 0.9, 0.999)));
		System.out.println("Initialized network.");

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information prior to file reading:");
		System.out.printf("Total Memory: %.3fMB%n", tMem / (1024.0 * 1024.0));
		System.out.printf("Free Memory: %.3fMB", fMem / (1024.0 * 1024.0));
		System.out.println();

		String pathTest = "/Users/edwincarlsson/Downloads/mnist/mnist_test.csv";
		String pathTrain = "/Users/edwincarlsson/Downloads/mnist/mnist_train.csv";
		String base = "/Users/edwincarlsson/Downloads/mnist/";

		imagesTrain = generateDataFromCSV(pathTrain);
		imagesValidate = generateDataFromCSV(pathTest);

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information after reading files:");
		System.out.printf("Total Memory: %.3fMB\n", tMem / (1024.0 * 1024.0));
		System.out.printf("Free Memory: %.3fMB\n", fMem / (1024.0 * 1024.0));
		System.out.println();

		Collections.shuffle(imagesTrain);
		Collections.shuffle(imagesValidate);

		final List<NetworkInput> imagesTest = imagesTrain.subList(0, (int) (imagesTrain.size() * 0.1));

		imagesTest.addAll(imagesValidate.subList(0, (int) (imagesValidate.size() * 0.1)));

		System.out.println("Starting SGD...");
		network.trainWithMetrics(imagesTrain, imagesValidate, epochs, batch, true, base);
		System.out.println("Finished SGD!");
		System.out.println();
		System.out.println("Evaluating the test data.");
		double correct = network.evaluateTestData(imagesTest, 100);
		System.out.println("Correct evaluation percentage: " + correct + ".");
		System.out.println("Writing charts. Serialising.");
		network.writeObject(base);
	}

	protected static List<NetworkInput> generateDataFromCSV(final String path) throws IOException {
		return Files.lines(Paths.get(path)).map(e -> e.split(",")).map(NetworkUtilities::MNISTApply)
				.collect(Collectors.toList());
	}
}
