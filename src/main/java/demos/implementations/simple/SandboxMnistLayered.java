package demos.implementations.simple;

import demos.AbstractDemo;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.costfunctions.SmoothL1CostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.linearalgebra.simple.SMatrix;
import math.linearalgebra.simple.SimpleMatrix;
import math.optimizers.ADAM;
import neuralnetwork.DeepLearnable;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.SimpleInitializer;
import neuralnetwork.inputs.NetworkInput;
import neuralnetwork.layer.LayeredNetworkBuilder;
import neuralnetwork.layer.LayeredNeuralNetwork;
import neuralnetwork.layer.NetworkLayer;
import utilities.serialise.serialisers.SimpleSerializer;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxMnistLayered extends AbstractDemo<SMatrix> {

	@Override
	protected String outputDirectory() {
		return "E:\\Downloads\\serial_network";
	}

	@Override
	protected TrainingMethod networkTrainingMethod() {
		return TrainingMethod.NORMAL;
	}

	@Override
	protected Pair<Integer, Integer> epochBatch() {
		return Pair.of(8, 64);
	}

	@Override
	protected Triple<List<NetworkInput<SMatrix>>, List<NetworkInput<SMatrix>>, List<NetworkInput<SMatrix>>> getData() {
		String test = "\\mnist_test.csv";
		String train = "\\mnist_train.csv";
		String path = "E:\\Downloads\\archive";

		try (var trainInData = Files.lines(Paths.get(path + train));
				var testInData = Files.lines(Paths.get(path + test))) {
			List<NetworkInput<SMatrix>> trainData = trainInData.map(this::toMnist).collect(Collectors.toList());
			List<NetworkInput<SMatrix>> testData = testInData.map(this::toMnist).collect(Collectors.toList());
			int totalSize = trainData.size();
			int splitIndex = (int) (totalSize * 0.75);

			Collections.shuffle(trainData);
			List<NetworkInput<SMatrix>> trainingData = trainData.subList(0, splitIndex);
			List<NetworkInput<SMatrix>> validateData = trainData.subList(splitIndex, trainData.size());

			return Triple.of(trainingData, validateData, testData);

		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	protected DeepLearnable<SMatrix> createNetwork() {
		LeakyReluFunction<SMatrix> f = new LeakyReluFunction<>(0.01);
		var softMax = new SoftmaxFunction<SMatrix>();
		var b = new LayeredNetworkBuilder<SMatrix>(28 * 28).layer(new NetworkLayer<>(f, 784, 0.1))
				.layer(new NetworkLayer<>(f, 30, 0.1)).layer(new NetworkLayer<>(f, 70, 0.1))
				.layer(new NetworkLayer<>(softMax, 10)).clipping(true).costFunction(new SmoothL1CostFunction<>())
				.evaluationFunction(new ArgMaxEvaluationFunction<>()).optimizer(new ADAM<>(0.01, 0.9, 0.999))
				.clipping(true).initializer(new SimpleInitializer(MethodConstants.XAVIER, MethodConstants.SCALAR));
		return b.create();
	}

	@Override
	protected void serialise(DeepLearnable<SMatrix> in) {
		SimpleSerializer layeredSerializer = new SimpleSerializer();
		LayeredNeuralNetwork<SMatrix> actual = (LayeredNeuralNetwork<SMatrix>) in;
		layeredSerializer.serialize(new File(this.outputDirectory() + "\\Simple_Layered_MNIST_Network.json"), actual);
	}

	private NetworkInput<SMatrix> toMnist(String toMnist) {
		int imageSize = 28 * 28;
		int labelSize = 10;
		String labelString = toMnist.substring(0, 2).split(",")[0];
		String[] rest = toMnist.substring(2).split(",");

		int label = Integer.parseInt(labelString);
		double[] labels = new double[labelSize];
		labels[label] = 1;

		double[] values = new double[rest.length];

		for (int i = 0; i < imageSize; i++) {
			values[i] = Double.parseDouble(rest[i]) / 255;
		}

		return new NetworkInput<SMatrix>(new SimpleMatrix(values), new SimpleMatrix(labels));
	}
}
