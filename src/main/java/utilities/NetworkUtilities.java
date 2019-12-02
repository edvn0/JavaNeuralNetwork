package utilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import neuralnetwork.NetworkInput;
import org.ujmp.core.Matrix;

public class NetworkUtilities {

	public static List<NetworkInput> importFromInputStream(String path, int size, int offset)
		throws IOException {

		List<NetworkInput> fromStream;
		Stream<String> stream = Files.lines(Paths.get(path));

		fromStream = stream
			.skip(offset)
			.limit(size)
			.map(line -> line.split(","))
			.map(e -> constructDataFromDoubleArray(normalizeData(e)))
			.collect(Collectors.toList());

		return fromStream;
	}

	public static List<NetworkInput> importFromInputStream(Stream<String> path, int size,
		int offset)
		throws IOException {

		List<NetworkInput> fromStream;

		fromStream = path
			.limit(size)
			.skip(offset)
			.map(line -> line.split(","))
			.map(NetworkUtilities::apply)
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

	public static List<NetworkInput> importFromInputStream(final Stream<String> test, int size)
		throws IOException {
		return importFromInputStream(test, size, 0);
	}

	private static NetworkInput apply(String[] e) {
		return constructDataFromDoubleArray(normalizeData(e));
	}
}
