package demos;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import javax.imageio.ImageIO;
import math.activations.LeakyReluFunction;
import math.activations.SoftmaxFunction;
import math.activations.TanhFunction;
import math.error_functions.MeanSquaredCostFunction;
import math.evaluation.ThreshHoldEvaluationFunction;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import optimizers.ADAM;
import org.ujmp.core.Matrix;
import org.ujmp.core.Matrix;

public class MnistAutoEncoder {

	private static List<NetworkInput> imagesTrain;
	private static List<NetworkInput> imagesValidate;

	public static void main(String[] args) throws IOException {
		long tMem, fMem;

		NeuralNetwork network = new NeuralNetwork(
				new NetworkBuilder(5).setFirstLayer(784).setLayer(35, new LeakyReluFunction(0.01))
						.setLayer(10, new SoftmaxFunction()).setLayer(35, new LeakyReluFunction(0.01))
						.setLastLayer(784, new TanhFunction()).setCostFunction(new MeanSquaredCostFunction())
						.setEvaluationFunction(new ThreshHoldEvaluationFunction(0.01))
						.setOptimizer(new ADAM(0.001, 0.9, 0.999)));
		network.display();

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information prior to file reading:");
		System.out.printf("Total Memory: %.3fMB%n", tMem / (1024.0 * 1024.0));
		System.out.printf("Free Memory: %.3fMB", fMem / (1024.0 * 1024.0));
		System.out.println();

		final boolean existsMac = Files
				.exists(Paths.get("/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv"));
		final boolean existsVM = Files.exists(Paths.get("/home/edwin98carlsson/mnist-in-csv/mnist_train.csv"));
		final boolean existsWindows;/*
									 * Files
									 * .exists(Paths.get("/home/edwin98carlsson/mnist-in-csv/mnist_train.csv"))
									 */

		String base = "";
		String pathTrain = null;
		String pathTest = null;
		if (existsMac && !existsVM) {
			pathTrain = "/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_train.csv";
			pathTest = "/Users/edwincarlsson/Downloads/mnist-in-csv/mnist_test.csv";
			base = "/Users/edwincarlsson/Downloads";
		} else if (!existsMac && existsVM) {
			pathTrain = "/home/edwin98carlsson/mnist-in-csv/mnist_train.csv";
			pathTest = "/home/edwin98carlsson/mnist-in-csv/mnist_test.csv";
			base = "/home/edwin98carlsson/";
		}

		imagesTrain = gdCSV(pathTrain);
		imagesValidate = gdCSV(pathTest);

		tMem = Runtime.getRuntime().totalMemory();
		fMem = Runtime.getRuntime().freeMemory();
		System.out.println();
		System.out.println("Memory information after reading files:");
		System.out.printf("Total Memory: %.3fMB\n", tMem / (1024.0 * 1024.0));
		System.out.printf("Free Memory: %.3fMB\n", fMem / (1024.0 * 1024.0));
		System.out.println();

		Collections.shuffle(imagesTrain);
		Collections.shuffle(imagesValidate);

		final List<NetworkInput> imagesTest = imagesTrain.subList(0, (int) (imagesTrain.size() * 0.1));

		imagesTest.addAll(imagesValidate.subList(0, (int) (imagesValidate.size() * 0.1)));

		System.out.println("Starting SGD...");
		network.trainWithMetrics(imagesTrain, imagesValidate, 5, 128, true, base);
		System.out.println("Finished SGD!");
		System.out.println();
		System.out.println("Evaluating the test data.");
		Matrix k = network.predict(imagesTest.get(ThreadLocalRandom.current().nextInt(imagesTest.size())).getData());
		showImage(k);
		double correct = network.evaluateTestData(imagesTest, 100);
		System.out.println("Correct evaluation percentage: " + correct + ".");
		System.out.println("Writing charts. Serialising.");
		network.writeObject(base);
	}

	private static List<NetworkInput> gdCSV(final String pathTrain) throws IOException {
		List<NetworkInput> l = new ArrayList<>();
		try (var reader = new BufferedReader(new FileReader(pathTrain))) {
			String row;
			while ((row = reader.readLine()) != null) {
				String[] k = row.split(",");
				double[][] d = new double[28 * 28][1];
				for (int i = 1; i < k.length; i++) {
					if (Double.parseDouble(k[i]) > 1) {
						d[i - 1][0] = 1;
					} else {
						d[i - 1][0] = 0;
					}
				}
				l.add(new NetworkInput(Matrix.Factory.importFromArray(d), Matrix.Factory.importFromArray(d)));
			}
		} catch (FileNotFoundException e) {
			System.out.println(e);
		}

		return l.subList(0, 5000);
	}

	private static void showImage(final Matrix k) throws IOException {
		BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_4BYTE_ABGR);
		double[][] data = k.toDoubleArray();
		for (int i = 0; i < img.getWidth(); i++) {
			for (int j = 0; j < img.getHeight(); j++) {
				Color c = new Color(((int) data[i][j]) * 255);
				img.setRGB(i, j, c.getRGB());
			}
		}

		ImageIO.write(img, "png", new File("/Users/edwincarlsson/Downloads"));
	}

}
