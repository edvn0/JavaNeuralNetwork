package utilities;

import static java.util.stream.Collectors.toList;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;
import neuralnetwork.inputs.NetworkInput;

public class NetworkUtilities {

	/**
	 * Maps a file input to {@link NetworkInput} objects to be used in the Neural Network
	 * <p>
	 * Splits every line in the file on a comma, and then applies the {@link Function} on that
	 * array.
	 * <p>
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
	public static <M> List<NetworkInput<M>> importFromInputPath(String path, int offset,
		Function<String[], NetworkInput<M>> f) throws IOException {
		try (var data = Files.lines(Path.of(path))) {
			return data.skip(1).map(line -> line.split(",")).map(f).collect(toList());
		}
	}

	public static <M> List<NetworkInput<M>> importFromInputStream(final Stream<String> test,
		int size,
		Function<String[], NetworkInput<M>> f) {
		return importFromInputStream(test, size, 0, f);
	}

	public static <M> List<NetworkInput<M>> importFromInputStream(Stream<String> path, int size,
		int offset,
		Function<String[], NetworkInput<M>> f) {
		return path.limit(size).skip(offset).map(line -> line.split(",")).map(f).collect(toList());
	}

	/**
	 * Splits the data into batches of training data.
	 *
	 * @param training  data to split
	 * @param batchSize by what batch size
	 *
	 * @return list of list of size batch with data
	 */
	public static <M> List<List<NetworkInput<M>>> batchSplitData(
		final List<NetworkInput<M>> training,
		final int batchSize) {
		List<List<NetworkInput<M>>> d = new ArrayList<>();
		for (int i = 0; i < training.size() - batchSize; i += batchSize) {
			d.add(training.subList(i, i + batchSize));
		}
		return d;
	}

	public static <M> List<Supplier<Stream<NetworkInput<M>>>> streamSplit(
		final List<NetworkInput<M>> training,
		final int batchSize) {
		List<Supplier<Stream<NetworkInput<M>>>> output = new ArrayList<>();
		for (int i = 0; i < training.size() - batchSize; i += batchSize) {
			final int index = i;
			final int nextIndex = index + batchSize;
			output.add(() -> training.subList(index, nextIndex).parallelStream());
		}
		return output;
	}

}
