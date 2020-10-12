package demos.implementations.ujmp;

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
import math.linearalgebra.ujmp.UJMPMatrix;
import math.optimizers.ADAM;
import neuralnetwork.DeepLearnable;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.MethodConstants;
import neuralnetwork.initialiser.UJMPInitialiser;
import neuralnetwork.inputs.NetworkInput;
import utilities.serialise.serialisers.UJMPSerializer;
import utilities.types.Pair;
import utilities.types.Triple;

public class SandboxMnist extends AbstractDemo<org.ujmp.core.Matrix> {

	@Override
	protected String outputDirectory() {
		return "/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/resources/mnist";
	}

	@Override
	protected TrainingMethod networkTrainingMethod() {
		return TrainingMethod.METRICS;
	}

	@Override
	protected Pair<Integer, Integer> epochBatch() {
		return Pair.of(9, 128);
	}

	@Override
	protected Triple<List<NetworkInput<org.ujmp.core.Matrix>>, List<NetworkInput<org.ujmp.core.Matrix>>, List<NetworkInput<org.ujmp.core.Matrix>>> getData() {
		String test = "/mnist_test.csv";
		String train = "/mnist_train.csv";
		String path = "/Volumes/Toshiba 1,5TB/mnist";

		try (var trainInData = Files.lines(Paths.get(path + train));
			var testInData = Files.lines(Paths.get(path + test))) {
			List<NetworkInput<org.ujmp.core.Matrix>> trainData = trainInData.map(this::toMnist)
				.collect(Collectors.toList());
			List<NetworkInput<org.ujmp.core.Matrix>> testData = testInData.map(this::toMnist)
				.collect(Collectors.toList());
			int totalSize = trainData.size();
			int splitIndex = (int) (totalSize * 0.75);

			Collections.shuffle(trainData);
			List<NetworkInput<org.ujmp.core.Matrix>> trainingData = trainData
				.subList(0, splitIndex);
			List<NetworkInput<org.ujmp.core.Matrix>> validateData = trainData
				.subList(splitIndex, trainData.size());

			return Triple.of(trainingData, validateData, testData);

		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	protected NeuralNetwork<org.ujmp.core.Matrix> createNetwork() {
		var f = new LeakyReluFunction<org.ujmp.core.Matrix>(0.01);
		return new NeuralNetwork<>(
			new NetworkBuilder<org.ujmp.core.Matrix>(4).setFirstLayer(784).setLayer(10, f)
				.setLayer(10, f)
				.setLastLayer(10, new SoftmaxFunction<>())
				.setCostFunction(new CrossEntropyCostFunction<>())
				.setEvaluationFunction(new ArgMaxEvaluationFunction<>())
				.setOptimizer(new ADAM<>(0.01, 0.9, 0.999)), // new ADAM<>(0.01, 0.9, 0.999)),
			new UJMPInitialiser(MethodConstants.XAVIER, MethodConstants.SCALAR));
	}

	@Override
	protected void serialise(DeepLearnable<org.ujmp.core.Matrix> in) {
		UJMPSerializer serializer = new UJMPSerializer();
		var out = (NeuralNetwork<org.ujmp.core.Matrix>) in;
		serializer.serialise(new File(this.outputDirectory() + "/UJMP_Mnist_Network.json"), out);
	}

	private NetworkInput<org.ujmp.core.Matrix> toMnist(String toMnist) {
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

		return new NetworkInput<org.ujmp.core.Matrix>(new UJMPMatrix(values),
			new UJMPMatrix(labels));
	}

}
