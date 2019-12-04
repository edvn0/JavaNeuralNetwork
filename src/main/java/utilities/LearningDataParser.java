package utilities;

import java.util.List;
import java.util.function.Function;
import neuralnetwork.NetworkInput;

public interface LearningDataParser {

	/**
	 * A parser which turns a list of strings into a NeuralNetwork-feedable data type.
	 *
	 * @param in        a list of strings
	 * @param size      how much of {@param in} should we use?
	 * @param offset    should we offset the data?
	 * @param f         a function to map a String in your dataset to a {@link NetworkInput}
	 * @param delimiter what String splits your data?
	 *
	 * @return a list of objects amenable in the Neural network
	 */
	List<NetworkInput> parseStringData(List<String> in, Function<String[], NetworkInput> f,
		int size, int offset, String delimiter);


}
