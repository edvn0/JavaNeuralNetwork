import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import math.ActivationFunction;
import math.ErrorFunction;
import math.MeanSquaredErrorFunction;
import math.TanhFunction;
import matrix.Matrix;
import neuralnetwork.NeuralNetwork;
import utilities.MatrixUtilities;

public class MNISTTester {

	public static Matrix[] imagesTrain = new Matrix[60000];
	public static Matrix[] imagesTest = new Matrix[10000];

	public static void main(String[] args) throws IOException {
		ActivationFunction[] functions = new ActivationFunction[6];
		functions[0] = new TanhFunction();
		functions[1] = new TanhFunction();
		functions[2] = new TanhFunction();
		functions[3] = new TanhFunction();
		functions[4] = new TanhFunction();
		functions[5] = new TanhFunction();
		ErrorFunction function = new MeanSquaredErrorFunction();
		NeuralNetwork network = new NeuralNetwork(10e-4, functions, function,
			new int[]{784, 2500, 2000, 1500, 1000, 10});
		network.initialiseFunctions();

		generateMnistDataFromCSV();

		List<Matrix[]> trainingdata = new ArrayList<>();
		for (int i = 0; i < imagesTrain.length; i++) {
			Matrix[] trainData = getTrainData(imagesTrain[i]);
			trainingdata.add(trainData);
		}

		int k = 0;

		network.train(trainingdata, "SGD");
		System.out.println(k);
		System.out.println("Done!");

		int correct = 0;
		k = 0;
		for (Matrix m : imagesTest) {
			Matrix[] testData = getTrainData(m);
			int index = MatrixUtilities.networkOutputsMax(network.predict(testData[0]));
			int label = getLabel(testData[1].getData());
			if (index == label) {
				correct++;
			}
			k++;
			if (k % 100 == 0) {
				System.out.println(
					"Image " + k + " was just now processed. There are " + (imagesTest.length - k)
						+ " left.");
			}
		}
		System.out.println((correct + 0d) / imagesTest.length);
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
				corr[0][0] = 0;
				break;
			case 1:
				corr[1][0] = 1 / 10d;
				break;
			case 2:
				corr[2][0] = 2 / 10d;
				break;
			case 3:
				corr[3][0] = 3 / 10d;
				break;
			case 4:
				corr[4][0] = 4 / 10d;
				break;
			case 5:
				corr[5][0] = 5 / 10d;
				break;
			case 6:
				corr[6][0] = 6 / 10d;
				break;
			case 7:
				corr[7][0] = 7 / 10d;
				break;
			case 8:
				corr[8][0] = 8 / 10d;
				break;
			case 9:
				corr[9][0] = 9 / 10d;
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
		System.out.println(Arrays.toString(mData));
		return mData;
	}

	public static void predictTestSet() {

	}

	public static void generateMnistDataFromCSV() throws IOException {
		List<String> trainData = Files.readAllLines(Paths.get(
			"/Users/edwincarlsson/Documents/Programmering/GradleProjects/NeuralNetwork/src/test/resources/mnist_train.csv"));
		List<String> testData = Files.readAllLines(Paths.get(
			"/Users/edwincarlsson/Documents/Programmering/GradleProjects/NeuralNetwork/src/test/resources/mnist_test.csv"));

		for (int i = 0; i < 60000; i++) {
			double[][] data = new double[1 + 28 * 28][1];
			String[] sp = trainData.get(i).split(",");
			changeData(data, sp);
			imagesTrain[i] = new Matrix(data);
		}

		for (int i = 0; i < imagesTest.length; i++) {
			double[][] data = new double[1 + 28 * 28][1];
			String[] sp = testData.get(i).split(",");
			changeData(data, sp);
			imagesTest[i] = new Matrix(data);
		}

	}

	private static void changeData(double[][] data, String[] sp) {
		for (int i = 1; i < data.length; i++) {
			data[i][0] = Integer.parseInt(sp[i]) / 255d;
		}
		data[0][0] = Integer.parseInt(sp[0]);
	}

}
