package demos.implementations.ojalgo;

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
import math.costfunctions.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.optimizers.ADAM;
import neuralnetwork.DeepLearnable;
import neuralnetwork.LayeredNetworkBuilder;
import neuralnetwork.LayeredNeuralNetwork;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.OjAlgoInitialiser;
import neuralnetwork.inputs.NetworkInput;
import neuralnetwork.layer.NetworkLayer;
import org.ojalgo.matrix.Primitive64Matrix;
import utilities.serialise.serialisers.OjAlgoLayeredSerializer;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxMnistLayered extends AbstractDemo<Primitive64Matrix> {

	@Override
	protected String outputDirectory() {
		return "/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/resources/mnist";
	}

	@Override
	protected TrainingMethod networkTrainingMethod() {
		return TrainingMethod.NORMAL;
	}

	@Override
	protected Pair<Integer, Integer> epochBatch() {
		return Pair.of(2, 64);
	}

	@Override
	protected Triple<List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>, List<NetworkInput<Primitive64Matrix>>> getData() {
		String test = "/mnist_test.csv";
		String train = "/mnist_train.csv";
		String path = "/Volumes/Toshiba 1,5TB/mnist";

		try (var trainInData = Files.lines(Paths.get(path + train));
			var testInData = Files.lines(Paths.get(path + test))) {
			List<NetworkInput<Primitive64Matrix>> trainData = trainInData.map(this::toMnist)
				.collect(Collectors.toList());
			List<NetworkInput<Primitive64Matrix>> testData = testInData.map(this::toMnist)
				.collect(Collectors.toList());
			int totalSize = trainData.size();
			int splitIndex = (int) (totalSize * 0.75);

			Collections.shuffle(trainData);
			List<NetworkInput<Primitive64Matrix>> trainingData = trainData.subList(0, splitIndex);
			List<NetworkInput<Primitive64Matrix>> validateData = trainData
				.subList(splitIndex, trainData.size());

			return Triple.of(trainingData, validateData, testData);

		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	protected DeepLearnable<Primitive64Matrix> createNetwork() {
		LeakyReluFunction<Primitive64Matrix> f = new LeakyReluFunction<>(0.01);
		var softMax = new SoftmaxFunction<Primitive64Matrix>();
		var b = new LayeredNetworkBuilder<Primitive64Matrix>(28 * 28)
			.layer(new NetworkLayer<>(f, 784))
			.layer(new NetworkLayer<>(f, 30))
			.layer(new NetworkLayer<>(f, 70))
			.layer(new NetworkLayer<>(f, 90))
			.layer(new NetworkLayer<>(f, 50))
			.layer(new NetworkLayer<>(softMax, 10))
			.costFunction(new CrossEntropyCostFunction<>())
			.evaluationFunction(new ArgMaxEvaluationFunction<>())
			.optimizer(new ADAM<>(0.001, 0.9, 0.999))
			.initializer(new OjAlgoInitialiser(MethodConstants.XAVIER, MethodConstants.SCALAR));
		return b.create();
	}

	@Override
	protected void serialise(DeepLearnable<Primitive64Matrix> in) {
		OjAlgoLayeredSerializer layeredSerializer = new OjAlgoLayeredSerializer();
		LayeredNeuralNetwork<Primitive64Matrix> actual = (LayeredNeuralNetwork<Primitive64Matrix>) in;
		layeredSerializer
			.serialise(new File(this.outputDirectory() + "/OjAlgo_XOR_Network.json"), actual);
	}

	private NetworkInput<Primitive64Matrix> toMnist(String toMnist) {
		int imageSize = 28 * 28;
		int labelSize = 10;
		String labelString = toMnist.substring(0, 2).split(",")[0];
		String[] rest = toMnist.substring(2, toMnist.length()).split(",");

		int label = Integer.parseInt(labelString);
		double[] labels = new double[labelSize];
		labels[label] = 1;

		double[] values = new double[rest.length];

		for (int i = 0; i < imageSize; i++) {
			values[i] = Double.parseDouble(rest[i]) / 255;
		}

		return new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(values),
			new OjAlgoMatrix(labels));
	}
}
