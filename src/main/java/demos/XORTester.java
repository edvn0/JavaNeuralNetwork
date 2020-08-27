package demos;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.imageio.ImageIO;
import math.activations.SigmoidFunction;
import math.activations.TanhFunction;
import math.error_functions.MeanSquaredCostFunction;
import math.evaluation.ThreshHoldEvaluationFunction;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NetworkInput;
import neuralnetwork.NeuralNetwork;
import optimizers.StochasticGradientDescent;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class XORTester {

	private List<NetworkInput> data;

	private BufferedImage[] images;

	private double[][] xorData = new double[][] { { 0, 1 }, { 0, 0 }, { 1, 1 }, { 1, 0 } };
	private double[][] xorLabel = new double[][] { { 1 }, { 0 }, { 0 }, { 1 } };

	private NeuralNetwork network;
	private String path;

	private int imagesSize;
	private int w, h;

	XORTester(String path, int size) {
		this.path = path;
		this.imagesSize = size;
		w = 600;
		h = 600;
		images = new BufferedImage[imagesSize];

		data = new ArrayList<>();
		SecureRandom r = new SecureRandom();
		for (int i = 0; i < 10000; i++) {
			double[][] cData;
			double[][] cLabel;
			int rd = r.nextInt(xorData.length);
			cData = new double[][] { xorData[rd] };
			cLabel = new double[][] { xorLabel[rd] };
			data.add(new NetworkInput(Matrix.Factory.importFromArray(cData).transpose(),
					Matrix.Factory.importFromArray(cLabel).transpose()));
		}
		Collections.shuffle(data);

		network = new NeuralNetwork(
				new NetworkBuilder(4).setFirstLayer(2).setLayer(10, new TanhFunction()).setLayer(10, new TanhFunction())
						.setLastLayer(1, new SigmoidFunction()).setCostFunction(new MeanSquaredCostFunction())
						.setEvaluationFunction(new ThreshHoldEvaluationFunction(0.025))
						.setOptimizer(new StochasticGradientDescent(0.05)));

		network.display();
		network.train(data.subList(0, 1000), data.subList(1000, 2000), 70, 64);
	}

	private void run(int in) throws IOException {
		if (w % in != 0 && h % in != 0) {
			throw new IllegalArgumentException();
		}
		int cols = w / in;
		int rows = h / in;
		for (int l = 0; l < imagesSize; l++) {
			BufferedImage img = new BufferedImage(600, 600, BufferedImage.TYPE_INT_ARGB);
			images[l] = img;
			System.out.println("Image " + l);
			for (int i = 0; i < cols; i++) {
				for (int j = 0; j < rows; j++) {
					double col = (double) i / cols;
					double row = (double) j / rows;
					double out = 0;
					if (i % (in) == 0) {
						out = network.predict(toInputMatrix(col, row)).doubleValue();
					}
					out *= 255;
					int colors = (int) out;
					Color c = new Color(colors, colors, colors);
					img.setRGB(i, j, c.getRGB());
				}
			}
		}

		int k = 0;
		for (BufferedImage img : images) {
			String out = path + "/XOR_" + k++ + ".png";
			ImageIO.write(img, "png", new File(out));
		}
	}

	private Matrix toInputMatrix(final double col, final double row) {
		return Matrix.Factory.importFromArray(new double[][] { { (double) col }, { (double) row } });
	}

	public static void main(String[] args) throws IOException {
		new XORTester("C:\\Users\\edvin\\Downloads", 3).run(1);
	}
}