package demos;

import java.io.IOException;
import java.util.List;
import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import org.ujmp.core.Matrix;
import utilities.NetworkUtilities;

public class IrisPrediction {

	public static void main(String[] args) throws IOException {
		ActivationFunction[] activationFunctions = new ActivationFunction[4];
		activationFunctions[0] = new ReluFunction();
		activationFunctions[1] = new ReluFunction();
		activationFunctions[2] = new ReluFunction();
		activationFunctions[3] = new SoftmaxFunction();
		ErrorFunction errorFunction = new CrossEntropyErrorFunction();
		EvaluationFunction evaluationFunction = new MnistEvaluationFunction();
		NeuralNetwork network = new NeuralNetwork(0.01, activationFunctions, errorFunction,
			evaluationFunction, new int[]{4, 100, 100, 3}, , , );

		List<NetworkInput> inputList = NetworkUtilities.importFromInputPath(
			"/Users/edwincarlsson/Downloads/DataAnalysisKaggle/scaled_iris_correct.csv", 1,
			IrisPrediction::toDenseMatrix);

		network
			.stochasticGradientDescent(inputList.subList(0, 100), inputList.subList(101, 150), 900,
				8);
	}

	public static int[] toSkip = {1, 1, 0, 0, 0, 0, 0};
	public static String[] flowers = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

	public static NetworkInput toDenseMatrix(String[] in) {
		String[] copy = new String[in.length];
		int k = 0;
		for (String s : in) {
			copy[k++] = s.replace("\"", "");
		}

		double[][] dataDouble = new double[4][1];
		double[][] labelDouble = new double[3][1];
		for (int i = 2; i < toSkip.length - 1; i++) {
			dataDouble[i - 2][0] = Double.parseDouble(copy[i]);
		}

		int flowerIndex = -1;
		for (int i = 0; i < flowers.length; i++) {
			if (copy[copy.length - 1].equals(flowers[i])) {
				flowerIndex = i;
				break;
			}
		}

		labelDouble[flowerIndex][0] = 1;

		return new NetworkInput(Matrix.Factory.importFromArray(dataDouble),
			Matrix.Factory.importFromArray(labelDouble));

	}

}
