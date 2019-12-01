package neuralnetwork;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileSystemView;
import math.activations.ActivationFunction;
import math.activations.ReluFunction;
import math.activations.SoftmaxFunction;
import math.errors.CrossEntropyErrorFunction;
import math.errors.ErrorFunction;
import math.evaluation.EvaluationFunction;
import math.evaluation.MnistEvaluationFunction;
import org.ujmp.core.Matrix;

public class MNISTTester {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesTest;

	public static void main(String[] args) throws IOException {
		NeuralNetwork network;

		JFileChooser file = new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());
		file.setDialogTitle("Choose the serialisation file for you network.");
		file.setFileSelectionMode(JFileChooser.FILES_ONLY);
		file.setFileFilter(new FileFilter() {
			@Override
			public boolean accept(final File f) {
				return f.getAbsoluteFile().toString().endsWith(".ser");
			}

			@Override
			public String getDescription() {
				return null;
			}
		});

		int ret = file.showOpenDialog(null);

		if (ret == JFileChooser.APPROVE_OPTION) {
			System.out.println("Read network.");
			File f = file.getSelectedFile();
			network = NeuralNetwork.readObject(f.getAbsolutePath());
		} else {
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
			network = new NeuralNetwork(0.041, functions, function, eval,
				new int[]{784, 2000, 1500, 1000, 500, 10});
		}

		/*System.out.println("Starting bGD");
		batchGradientDescentKindOf(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv", 10000, network);
		System.out.println("Ending sGD.");*/

		imagesTrain = generateDataFromCSV(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv");
		imagesTest = generateDataFromCSV(
			"/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_test.csv");

		System.out.println("Starting SGD...");
		network.stochasticGradientDescent(imagesTrain, imagesTest, 10, 64);
		System.out.println("Finished SGD!");
		network.outputChart("/Users/edwincarlsson/Desktop");
		network.writeObject("/Users/edwincarlsson/Desktop");
	}

	private static List<NetworkInput> generateDataFromCSV(String path) throws IOException {
		List<NetworkInput> things = new ArrayList<>();

		Files.readAllLines(Paths.get(path))
			.forEach((e) -> things.add(getTrainData(normalizeData(e.split(",")))));

		return things;
	}

	private static double[][] normalizeData(String[] split) {
		double[][] d = new double[1 + 28 * 28][1];
		for (int i = 1; i < split.length; i++) {
			if (Double.parseDouble(split[i]) > 1) {
				d[i][0] = 0;
			} else {
				d[i][0] = 1;
			}
		}
		d[0][0] = Integer.parseInt(split[0]);
		return d;
	}

	private static NetworkInput getTrainData(double[][] in) {
		double[][] corr = new double[10][1];
		String num = String.valueOf(in[0][0]);
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
			data[j - 1][0] = in[j][0];
		}
		return new NetworkInput(Matrix.Factory.importFromArray(data),
			Matrix.Factory.importFromArray(corr));
	}

}
