package demos;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SigmoidFunction;
import math.errors.BinaryCrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import utilities.NetworkUtilities;

public class CreditCardPrediction {

	public static void main(String[] args) throws IOException {
		List<NetworkInput> inputs = NetworkUtilities.importFromInputPath(
			"/Users/edwincarlsson/Downloads/DataAnalysisKaggle/scaled_theft.csv", 1,
			CreditCardPrediction::toDenseMatrix);

		Collections.shuffle(inputs);

		ActivationFunction[] aFunctions = new ActivationFunction[3];
		aFunctions[0] = new ReluFunction();
		aFunctions[1] = new ReluFunction();
		aFunctions[2] = new SigmoidFunction();
		ErrorFunction errorFunction = new BinaryCrossEntropyErrorFunction();
		EvaluationFunction evaluationFunction = new FraudEvaluationFunction();
		NeuralNetwork network = new NeuralNetwork(0.00035, aFunctions, errorFunction,
			evaluationFunction, new int[]{30, 25, 1}, , , );

		System.out.println("initializing SGD");
		network.stochasticGradientDescent(inputs, inputs, 100, 32);
		System.out.println("finished sgd");

		network.outputChart("/Users/edwincarlsson/Downloads/DataAnalysisKaggle");
		network.writeObject("/Users/edwincarlsson/Downloads/DataAnalysisKaggle");


	}

	private static NetworkInput toDenseMatrix(final String[] strings) {
		DenseMatrix data;
		DenseMatrix label;

		double[][] dataD = new double[30][1];
		double[][] labelD = new double[1][1];
		for (int i = 0; i < dataD.length; i++) {
			dataD[i][0] = Double.parseDouble(strings[i + 1]);
		}
		dataD[dataD.length - 2][0] = dataD[dataD.length - 2][0] / 172792.0;

		labelD[0][0] = Double.parseDouble(strings[strings.length - 1]);

		data = Matrix.Factory.importFromArray(dataD);
		label = Matrix.Factory.importFromArray(labelD);

		return new NetworkInput(data, label);
	}

}
