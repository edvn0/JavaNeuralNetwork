package utilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import neuralnetwork.NetworkInput;
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
			.collect(Collectors.toList());

		stream.close();

		return fromStream;
	}

	public static List<NetworkInput> importFromInputStream(Stream<String> path, int size,
		int offset,
		Function<String[], NetworkInput> f) {

		List<NetworkInput> fromStream;

		fromStream = path.limit(size).skip(offset).map(line -> line.split(",")).map(f)
			.collect(Collectors.toList());

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
}
