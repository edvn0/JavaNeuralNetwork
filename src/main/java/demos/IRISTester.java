package demos;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyCostFunction;
import math.evaluation.ArgMaxEvaluationFunction;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import optimizers.ADAM;
import utilities.NetworkUtilities;
import utilities.data.DataImport;
import utilities.data.DataImport.DataMethod;

public class IRISTester {

	public static void main(String[] args) throws IOException {
		NeuralNetwork network = new NeuralNetwork(new NetworkBuilder(3)
			.setFirstLayer(4)
			.setLayer(100, new LeakyReluFunction(0.01))
			.setLastLayer(3, new SoftmaxFunction())
			.setEvaluationFunction(new ArgMaxEvaluationFunction())
			.setCostFunction(new CrossEntropyCostFunction())
			.setOptimizer(new ADAM(0.001, 0.9, 0.999))
		);

		String in = "/Users/edwincarlsson/Downloads/iris.csv";
		List<NetworkInput> fullDataSet = NetworkUtilities
			.importData(in,
				new DataImport()
					.setDataSize(4)
					.setLabelSize(3)
					.setImportMethod(DataMethod.MINMAX)
					.setMinMaxArray(new double[][]{{1, 2}, {0.01, 1}, {1, 3}, {3, 10}})
					.compile(),
				",");

		Collections.shuffle(fullDataSet);
		List<NetworkInput> training = fullDataSet.subList(0, 110);
		List<NetworkInput> validation = fullDataSet.subList(110, 130);
		List<NetworkInput> testing = fullDataSet.subList(130, 150);

		network.trainWithMetrics(training,
			validation,
			70,
			1, true, "/Users/edwincarlsson/Downloads");
		System.out.println(network.evaluateTestData(testing, 1000));
	}
}
