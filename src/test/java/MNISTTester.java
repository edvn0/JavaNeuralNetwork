import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import math.activations.ActivationFunction;
import math.activations.SigmoidFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;
import matrix.Matrix;
import neuralnetwork.NeuralNetwork;
import org.jetbrains.annotations.NotNull;

public class MNISTTester {

	private static List<Matrix[]> imagesTrain;
	private static List<Matrix[]> imagesTest;

	public static void main(String[] args) throws IOException {
		ActivationFunction[] functions = new ActivationFunction[3];
		functions[0] = new SigmoidFunction();
		functions[1] = new SigmoidFunction();
		functions[2] = new SoftmaxFunction();
		ErrorFunction function = new CrossEntropyErrorFunction();
		EvaluationFunction eval = new MnistEvaluationFunction();
		NeuralNetwork network = new NeuralNetwork(5e-3, functions, function, eval,
			new int[]{784, 100, 10});

		imagesTrain = generateDataFromCSV(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv");
		imagesTest = generateDataFromCSV(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_test.csv");

		System.out.println("Starting SGD...");
		network.stochasticGradientDescent(imagesTrain, imagesTest, 30, 32);
		System.out.println("Finished SGD!");
	}

	private static List<Matrix[]> generateDataFromCSV(String path) throws IOException {
		List<Matrix[]> things = new ArrayList<>();

		Files.readAllLines(Paths.get(path))
			.forEach((e) -> things.add(getTrainData(new Matrix(normalizeData(e.split(","))))));

		return things;
	}

	private static double[][] normalizeData(String[] split) {
		double[][] d = new double[1 + 28 * 28][1];
		for (int i = 1; i < split.length; i++) {
			d[i][0] = Double.parseDouble(split[i]) / 255d;
		}
		d[0][0] = Integer.parseInt(split[0]);
		return d;
	}


	private static int getLabel(double[][] testDatum) {
		for (int i = 1; i < testDatum.length; i++) {
			if (testDatum[i][0] != 0) {
				return (int) testDatum[i][0];
			}
		}
		return 0;
	}

	private static Matrix[] getTrainData(Matrix m) {
		Matrix[] mData = new Matrix[2];
		double[][] corr = new double[10][1];
		String num = m.getData()[0][0] + "";
		String newNum = num.substring(0, 1);
		switch (Integer.parseInt(newNum)) {
			case 0:
				corr[0][0] = 1;
				break;
			case 1:
				corr[1][0] = 1;
				break;
			case 2:
				corr[2][0] = 1;
				break;
			case 3:
				corr[3][0] = 1;
				break;
			case 4:
				corr[4][0] = 1;
				break;
			case 5:
				corr[5][0] = 1;
				break;
			case 6:
				corr[6][0] = 1;
				break;
			case 7:
				corr[7][0] = 1;
				break;
			case 8:
				corr[8][0] = 1;
				break;
			case 9:
				corr[9][0] = 1;
				break;
			default:
				break;
		}

		double[][] data = new double[28 * 28][1];

		int dataSize = data.length;
		for (int j = 1; j < dataSize; j++) {
			data[j - 1][0] = m.getData()[j][0];
		}
		mData[0] = new Matrix(data);
		mData[1] = new Matrix(corr);
		return mData;
	}

	public static void predictTestSet() {

	}

	private static void normalizeData(@NotNull double[][] data, String[] sp) {
		for (int i = 1; i < data.length; i++) {
			data[i][0] = Integer.parseInt(sp[i]) / 255d;
		}
		data[0][0] = Integer.parseInt(sp[0]);
	}

}
