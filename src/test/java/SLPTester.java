import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import neuralnetwork.SingleLayerPerceptron;
import utilities.MatrixUtilities;

public class SLPTester {


	public static void main(String[] args) {
		SingleLayerPerceptron perceptron = new SingleLayerPerceptron(95 * 71, 10, 3, 0.11);
		perceptron.setDefaultValues();

		//SingleLayerPerceptron perceptron1 = builder
		//	.fromJson("/Users/edwincarlsson/Downloads/BulkResizePhotos/info.json",
		//		SingleLayerPerceptron.class);

		// rect (0) : {1,0,0}
		// circle (1) : {0,0,1}
		// tri (0.5d) : {0,1,0}
		BufferedImage[] images = readImages("/Users/edwincarlsson/Downloads/BulkResizePhotos");
		double[][] correct = {{1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {1, 0, 0}, {0, 0, 1}};

		for (int i = 0; i < 10000; i++) {
			perceptron.train(getGreyScalePixels(images[i % 6]), correct[i % 6]);
		}

		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[0])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[1])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[2])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[3])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[4])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[5])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[6])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[7])))
			.show();
		MatrixUtilities.networkOutputsSoftMax(perceptron.predict(getGreyScalePixels(images[8])))
			.show();

	}

	private static double[] getGreyScalePixels(BufferedImage image) {
		double[] inputs = new double[95 * 71];
		int k = 0;
		for (int i = 0; i < image.getWidth(); i++) {
			for (int j = 0; j < image.getHeight(); j++) {
				Color c = new Color(image.getRGB(i, j));
				double rgb = (c.getRed() + c.getGreen() + c.getBlue()) / 3d;
				if (rgb > 1) {
					inputs[k] = 1;
				} else {
					inputs[k] = 0;
				}
				k++;
			}
		}
		return inputs;
	}

	private static BufferedImage[] readImages(String s) {
		BufferedImage[] images = new BufferedImage[9];
		try {
			for (int i = 0; i < images.length; i++) {
				images[i] = ImageIO.read(new File(s + "/Shapes" + (i + 1) + ".jpg"));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return images;
	}

}
