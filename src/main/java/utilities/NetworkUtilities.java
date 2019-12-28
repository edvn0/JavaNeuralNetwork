package utilities;

import static java.util.stream.Collectors.toList;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;
import neuralnetwork.NetworkInput;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class NetworkUtilities {

	/**
	 * Maps a file input to {@link NetworkInput} objects to be used in the Neural Network
	 *
	 * Splits every line in the file on a comma, and then applies the {@link Function} on that
	 * array.
	 *
	 * Example: "1.1,1.3,0.13,0.01,TRUE"->["1.1","1.3",0.13","0.01","TRUE"]->
	 * Vector(1.1,1.3,0.13,0.01), Vector(0,0,1,0), if TRUE in some represents a one hot vector.
	 * Vector here is not some class in this project, rather a representation of what the function
	 * should do.
	 *
	 * @param path   Path to the file
	 * @param offset should a header line be skipped? Or even more lines?
	 * @param f      a map from a split string to a NetworkInput.
	 *
	 * @return a {@link List} with objects that can be used with the Neural Network
	 *
	 * @throws IOException if the file associated with the path does not exist.
	 */
	public static List<NetworkInput> importFromInputPath(String path, int offset,
		Function<String[], NetworkInput> f)
		throws IOException {

		List<NetworkInput> fromStream;
		Stream<String> stream = Files.lines(Paths.get(path));

		fromStream = stream.skip(offset).map(line -> line.split(",")).map(f)
			.collect(toList());

		stream.close();

		return fromStream;
	}

	public static List<NetworkInput> importFromPath(String path, DataImport di) throws IOException {

		Stream<String> stream = Files.lines(Paths.get(path));

		return stream.map(e -> e.split(",")).map(di.function).collect(toList());
	}

	public static List<NetworkInput> importFromInputStream(Stream<String> path, int size,
		int offset,
		Function<String[], NetworkInput> f) {

		List<NetworkInput> fromStream;

		fromStream = path.limit(size).skip(offset).map(line -> line.split(",")).map(f)
			.collect(toList());

		return fromStream;
	}

	private static double[][] normalizeData(String[] split) {
		double[][] d = new double[1 + 28 * 28][1];
		for (int i = 1; i < split.length; i++) {
			if (Double.parseDouble(split[i]) > 1) {
				d[i][0] = 1;
			} else {
				d[i][0] = 0;
			}
		}
		d[0][0] = Integer.parseInt(split[0]);
		return d;
	}

	private static NetworkInput constructDataFromDoubleArray(double[][] in) {
		double[][] corr = new double[10][1];
		int index = (int) in[0][0];
		corr[index][0] = 1;
		double[][] data = new double[28 * 28][1];

		int dataSize = data.length;
		for (int j = 1; j < dataSize; j++) {
			data[j - 1][0] = in[j][0];
		}
		return new NetworkInput(Matrix.Factory.importFromArray(data),
			Matrix.Factory.importFromArray(corr));
	}

	public static List<NetworkInput> importFromInputStream(final Stream<String> test, int size,
		Function<String[], NetworkInput> f) throws IOException {
		return importFromInputStream(test, size, 0, f);
	}

	public static NetworkInput MNISTApply(String[] e) {
		return constructDataFromDoubleArray(normalizeData(e));
	}

	public static List<List<NetworkInput>> splitData(final List<NetworkInput> training,
		final int batchSize) {
		List<List<NetworkInput>> d = new ArrayList<>();
		for (int i = 0; i < training.size() - batchSize; i += batchSize) {
			d.add(training.subList(i, i + batchSize));
		}
		return d;
	}


	public static List<NetworkInput> importData(String path, DataImport dI, String splitOn)
		throws IOException {

		Stream<String> lines = Files.lines(Paths.get(path));

		return lines.map(e -> e.split(splitOn)).map(dI.function)
			.collect(toList());
	}


	// TODO: Implement DataImport thingie
	public static class DataImport {

		private NetworkInput applyMinMax(String[] a) {
			double[][] data, label;
			data = new double[this.dataSize][1];
			label = new double[this.labelSize][1];

			for (int i = 0; i < dataSize; i++) {
				data[i][0] = minMaxNormalization(
					Double.parseDouble(a[i]), this.minMax[i][0], this.minMax[i][1]);
			}

			int oneHot = Integer.parseInt(a[a.length - 1]);
			label[oneHot][0] = 1;

			return new NetworkInput(DenseMatrix.Factory.importFromArray(data),
				DenseMatrix.Factory.importFromArray(label));
		}

		public enum DataMethod {

			MINMAX, Z_TRANSFORM, UNMODIFIED
		}

		boolean[] completed = new boolean[4];
		private int index;

		private Function<String[], NetworkInput> function;
		private int dataSize, labelSize;

		private double[][] minMax;
		private double[] means, standardDeviations;

		private DataMethod method;

		public DataImport() {
			index = 0;
		}

		public DataImport setMinMaxArray(double[][] minMax) {
			completed[index++] = true;
			if (minMax[0].length > 2) {
				throw new IllegalArgumentException(
					"Incorrect dimensions, only provide min and max for each row.");
			}
			this.minMax = minMax;
			return this;
		}

		public DataImport setMean(double[] mean) {
			this.completed[index++] = true;
			this.means = mean;
			return this;
		}

		public DataImport setStandardDeviation(double[] stddev) {
			this.completed[index++] = true;
			this.standardDeviations = stddev;
			return this;
		}

		public DataImport setDataSize(int k) {
			completed[index++] = true;
			this.dataSize = k;
			return this;
		}

		public DataImport setLabelSize(int k) {
			completed[index++] = true;
			this.labelSize = k;
			return this;
		}

		public DataImport setImportMethod(DataMethod m) {
			completed[index++] = true;
			this.method = m;
			return this;
		}

		public DataImport compile() {
			for (boolean b : completed) {
				if (!b) {
					throw new IllegalArgumentException(
						"The model is not yet ready for compilation.");
				}
			}

			switch (method) {
				case MINMAX:
					this.function = this::applyMinMax;
					break;
				case UNMODIFIED:
					this.function = this::applyUnmodified;
					break;
				case Z_TRANSFORM:
					this.function = this::applyZTransform;
					break;
				default:
					throw new RuntimeException("No method was provided, cannot compile the model.");
			}

			return this;
		}

		private NetworkInput applyUnmodified(final String[] strings) {
			double[][] data = new double[this.dataSize][1];
			double[][] label = new double[this.labelSize][1];

			for (int i = 0; i < strings.length - 1; i++) {
				data[i][0] = Double.parseDouble(strings[i]);
			}

			int index = Integer.parseInt(strings[strings.length - 1]);
			label[index][0] = 1;

			return new NetworkInput(DenseMatrix.Factory.importFromArray(data),
				DenseMatrix.Factory.importFromArray(label));
		}

		private NetworkInput applyZTransform(final String[] strings) {
			double[][] data = new double[this.dataSize][1];
			double[][] label = new double[this.labelSize][1];

			for (int i = 0; i < strings.length - 1; i++) {
				data[i][0] = zTransform(Double.parseDouble(strings[i]), i);
			}

			int index = Integer.parseInt(strings[strings.length - 1]);
			label[index][0] = 1;

			return new NetworkInput(DenseMatrix.Factory.importFromArray(data),
				DenseMatrix.Factory.importFromArray(label));
		}

		private double zTransform(final double parseDouble, int row) {
			return (parseDouble - means[row]) / (standardDeviations[row]);
		}

		public DataImport setMap(Function<String[], NetworkInput> e) {
			this.function = e;
			return this;
		}

		private static double minMaxNormalization(final double in, final double min,
			final double max) {
			return (in - min) / (max - min);
		}
	}
}
