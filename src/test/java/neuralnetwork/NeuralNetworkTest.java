package neuralnetwork;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import utilites.MatrixUtilities;

class NeuralNetworkTest {

	public static final String[] NAMES = new String[]{"Andreas", "Marie", "Mikael", "Stefan",
		"Ulf"};

	public static void main(String[] args) {
		setUp();
	}

	static void setUp() {
		NeuralNetwork network = new NeuralNetwork(2, 16, 5, 0.2);
		network.setDefaultValues();
		startTraining(network, 1000000,
			"/Users/edwincarlsson/Documents/Programmering/GradleProjects/NeuralNetwork/src/test/resources/People");

		BufferedImage[] peopleImages = getImages();
		for (int i = 0; i < peopleImages.length; i++) {
			double[] metrics = getMetrics(peopleImages[i], peopleImages[i].getWidth(),
				peopleImages[i].getHeight());
			System.out.println(Arrays.toString(metrics));
			int min = MatrixUtilities.networkOutputsMin(network.predict(metrics));
			System.out.println(NAMES[min]);
		}
	}

	private static BufferedImage[] getImages() {
		BufferedImage[] images = new BufferedImage[5];
		for (int i = 0; i < 5; i++) {
			int index = i + 1;
			try {
				images[i] = ImageIO.read(new File(
					"/Users/edwincarlsson/Documents/Programmering/GradleProjects/NeuralNetwork/src/test/resources/Test/1_"
						+ index + ".png"));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return images;
	}

	private static void startTraining(NeuralNetwork network, int iterations, String path) {
		double[][] metrics = initialiseMetrics(path,
			new String[]{"Andreas", "Marie", "Mikael", "Stefan", "Ulf"});
		for (int i = 0; i < iterations; i++) {
			double[][] knownOutput = new double[][]{{1d, 0d, 0d, 0d, 0d}, {0d, 1d, 0d, 0d, 0d},
				{0d, 0d, 1d, 0d, 0d}, {0d, 0d, 0d, 1d, 0d}, {0d, 0d, 0d, 0d, 1d}};
			network.trainNeuralNetwork(metrics[i % 5], knownOutput[i % 5]);
		}
	}

	/**
	 * Method to calculate the necessary statistical metrics (mean, standard deviation) for
	 * recognition
	 *
	 * @return array of double vectors (mean,standard deviation).
	 */
	private static double[][] initialiseMetrics(String path, String[] NAMES) {

		double[][] measurements = new double[NAMES.length][2];
		// One vector per name in NAMES.

		//Loop through each folder in people
		for (int numOfFolders = 0; numOfFolders < NAMES.length; numOfFolders++) {
			double folderImageMean = 0; // Mean of images in this folder
			double folderImageStandardDeviation = 0; // Stddev of images in this folder

			for (int i = 0; i < 9; i++) { //Loop through each image in each folder
				int index = i + 2; // file nr.2-10.
				String imgPath = path + "/" + NAMES[numOfFolders] + "/" + index + ".png";
				File file = new File(imgPath);
				try {
					BufferedImage image = ImageIO.read(file);
					double width = image.getWidth();
					double height = image.getHeight();

					double imageMean = calculateImageMean(image, width, height);
					double imageDeviation = calculateImageStandardDeviation(imageMean, image, width,
						height);

					folderImageMean += imageMean;
					folderImageStandardDeviation += imageDeviation;

				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			final int trainDataSize = 9; // Images in each folder.
			folderImageMean =
				folderImageMean / trainDataSize; // Normalize by how many images in this folder
			folderImageStandardDeviation = folderImageStandardDeviation / trainDataSize; // ^^
			measurements[numOfFolders][0] = folderImageMean;
			measurements[numOfFolders][1] = folderImageStandardDeviation;
		}
		return measurements;
	}

	/**
	 * Calculate image mean and stddev.
	 *
	 * @param img Input image
	 * @param width Input image width
	 * @param height Input image height
	 * @return {Mean, Standard Deviation}
	 */
	private static double[] getMetrics(BufferedImage img, int width, int height) {
		double mean = calculateImageMean(img, width, height);
		double standardDeviation = calculateImageStandardDeviation(mean, img, width, height);
		return new double[]{mean, standardDeviation};
	}

	private static double calculateImageMean(BufferedImage image, double width, double height) {
		double mean = 0;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				int rgb = image.getRGB(i, j);
				Color indexColor = new Color(rgb);
				int grey =
					(indexColor.getRed()
						+ indexColor.getGreen()
						+ indexColor.getBlue())
						/ 3; // Extra precaution if R!=B or B!=G or G!=R.
				mean += grey;
			}
		}
		return mean / (width * height);
	}

	private static double calculateImageStandardDeviation(double mean, BufferedImage image,
		double width,
		double height) {

		// LaTeX: $\sqrt{ \frac{1}{N} \sum_{i=0}^{k} (x_{i} - \text{mean})^{2}}$

		double stddev = 0;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				int rgb = image.getRGB(i, j);
				Color indexColor = new Color(rgb);
				int grey =
					(indexColor.getRed()
						+ indexColor.getGreen()
						+ indexColor.getBlue()) / 3; // Same precaution as in calculateMean
				double inner = Math.pow((grey - mean), 2);
				stddev += inner;
			}
		}
		stddev /= (width * height);
		return Math.sqrt(stddev);
	}
}