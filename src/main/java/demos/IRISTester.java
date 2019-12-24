package demos;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.NeuralNetwork.NetworkBuilder;
import optimizers.ADAM;
import org.ujmp.core.DenseMatrix;
import sun.nio.ch.Net;

public class IRISTester {

	public static void main(String[] args) throws IOException {
		NeuralNetwork network = new NeuralNetwork(new NetworkBuilder(3)
			.setFirstLayer(4)
			.setLayer(100, new LeakyReluFunction(0.01))
			.setLastLayer(3, new SoftmaxFunction())
			.setEvaluationFunction(new ArgMaxEvaluationFunction())
			.setOptimizer(new ADAM(0.001, 0.9, 0.999))
			.setCostFunction(new CrossEntropyCostFunction()));

		String in = "/Users/edwincarlsson/Downloads/iris.csv";
		List<NetworkInput> fullDataSet = getDataFromCSV(in);

		Collections.shuffle(fullDataSet);
		List<NetworkInput> training = fullDataSet.subList(0, 110);
		List<NetworkInput> validation = fullDataSet.subList(110, 130);
		List<NetworkInput> testing = fullDataSet.subList(130, 150);

		network.train(training,
			validation,
			10_000,
			1);
		System.out.println(network.evaluateTestData(testing, 1000));
		network.outputChart("/Users/edwincarlsson/Downloads");
	}

	// Summary Statistics:
	//	         	   Min  Max   Mean   SD   Class Correlation
	//   sepal length: 4.3  7.9   5.84  0.83    0.7826
	//    sepal width: 2.0  4.4   3.05  0.43   -0.4194
	//   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
	//    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
	private static List<NetworkInput> getDataFromCSV(String in) throws IOException {
		return Files.lines(Paths.get(in)).map(e -> e.split(",")).map(e -> {
			double[][] data, label;
			data = new double[4][1];
			label = new double[3][1];

			data[0][0] = minMaxNormalization(Double.parseDouble(e[0]), 4.3d, 7.9d);
			data[1][0] = minMaxNormalization(Double.parseDouble(e[1]), 2.0, 4.4d);
			data[2][0] = minMaxNormalization(Double.parseDouble(e[2]), 1, 6.9d);
			data[3][0] = minMaxNormalization(Double.parseDouble(e[3]), 0.1, 2.5d);

			int oneHot = Integer.parseInt(e[4]);
			label[oneHot][0] = 1;

			return new NetworkInput(DenseMatrix.Factory.importFromArray(data),
				DenseMatrix.Factory.importFromArray(label));
		}).collect(Collectors.toList());
	}

	private static double minMaxNormalization(final double in, final double min, final double max) {
		return (in - min) / (max - min);
	}

}
