package utilities;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import neuralnetwork.NetworkInput;

public class CsvDataParser implements LearningDataParser {

	@Override
	public List<NetworkInput> parseStringData(final List<String> in,
		final Function<String[], NetworkInput> f,
		final int size, final int offset, String delimiter) {
		return in.
			stream().
			skip(offset).
			limit(size).
			map(e -> e.split(delimiter)).
			map(f).
			collect(Collectors.toList());

	}

}
