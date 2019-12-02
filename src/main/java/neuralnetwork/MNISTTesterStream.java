package neuralnetwork;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Supplier;
import java.util.stream.Stream;
import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;

public class MNISTTesterStream {

	public static void main(String[] args) throws IOException {
		System.out.println("Initialized network.");
		ActivationFunction[] functions = new ActivationFunction[6];
		functions[0] = new ReluFunction();
		functions[1] = new ReluFunction();
		functions[2] = new ReluFunction();
		functions[3] = new ReluFunction();
		functions[4] = new ReluFunction();
		functions[5] = new SoftmaxFunction();
		ErrorFunction function = new CrossEntropyErrorFunction();
		EvaluationFunction eval = new MnistEvaluationFunction();
		NeuralNetwork network = new NeuralNetwork(0.0000015, functions, function, eval,
			new int[]{784, 2000, 1500, 1000, 500, 10});

		Supplier<Stream<String>> test = () -> {
			try {
				return Files.lines(Paths.get(
					"/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/java/neuralnetwork/mnist-in-csv/mnist_test.csv"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			return null;
		};
		Supplier<Stream<String>> train = () -> {
			try {
				return Files.lines(Paths.get(
					"/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/java/neuralnetwork/mnist-in-csv/mnist_train.csv"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			return null;
		};

		network.streamedSGD(train, test, 32, 10);


	}

}
