package utilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import neuralnetwork.NetworkInput;
import org.ujmp.core.Matrix;

public class NetworkUtilities {

	public static List<NetworkInput> importFromInputPath(String path, int offset,
		Function<String[], NetworkInput> f)
		throws IOException {

		List<NetworkInput> fromStream;
		Stream<String> stream = Files.lines(Paths.get(path));

		fromStream = stream
			.skip(offset)
			.map(line -> line.split(","))
			.map(f)
			.collect(Collectors.toList());

		return fromStream;
	}

	public static List<NetworkInput> importFromInputStream(Stream<String> path, int size,
		int offset, Function<String[], NetworkInput> f) {

		List<NetworkInput> fromStream;

		fromStream = path
			.limit(size)
			.skip(offset)
			.map(line -> line.split(","))
			.map(f)
			.collect(Collectors.toList());

		return fromStream;
	}

	public static double[][] normalizeData(String[] split) {
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

	public static NetworkInput constructDataFromDoubleArray(double[][] in) {
		double[][] corr = new double[10][1];
		String num = String.valueOf(in[0][0]);
		String newNum = num.substring(0, 1);
		int index = Integer.parseInt(newNum);
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
		Function<String[], NetworkInput> f)
		throws IOException {
		return importFromInputStream(test, size, 0, f);
	}

	public static NetworkInput apply(String[] e) {
		return constructDataFromDoubleArray(normalizeData(e));
	}
}
