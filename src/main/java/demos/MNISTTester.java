package demos;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import lombok.extern.slf4j.Slf4j;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.error_functions.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import optimizers.ADAM;
import utilities.NetworkUtilities;

@Slf4j
public class MNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesValidate;

	public static void main(final String[] args) {
		if (args.length != 3)
			throw new IllegalArgumentException("Need to supply epochs, batches, output path.");

		final int epochs = Integer.parseInt(args[0]);
		final int batch = Integer.parseInt(args[1]);
		final String output = args[3];

		NeuralNetwork network = NeuralNetwork.of(new NetworkBuilder(4).setFirstLayer(784)
				.setLayer(35, new LeakyReluFunction(0.01)).setLayer(35, new LeakyReluFunction(0.01))
				.setLastLayer(10, new SoftmaxFunction()).setCostFunction(new CrossEntropyCostFunction())
				.setEvaluationFunction(new ArgMaxEvaluationFunction()).setOptimizer(new ADAM(0.001, 0.9, 0.999)));

		imagesTrain = NetworkUtilities.readGzip("C:\\Users\\edvin\\Downloads\\train-images-idx3-ubyte.gz");
		imagesValidate = NetworkUtilities.readGzip("C:\\Users\\edvin\\Downloads\\t10k-images-idx3-ubyte.gz");

		Collections.shuffle(imagesTrain);
		Collections.shuffle(imagesValidate);

		final List<NetworkInput> imagesTest = imagesTrain.subList(0, (int) (imagesTrain.size() * 0.1));

		imagesTest.addAll(imagesValidate.subList(0, (int) (imagesValidate.size() * 0.1)));

		network.trainWithMetrics(imagesTrain, imagesValidate, epochs, batch, true, output);

		double correct = network.evaluateTestData(imagesTest, 100);
		log.info(String.format("Correct evaluation percentage: %d", correct));
		network.writeObject(output);
	}
}
