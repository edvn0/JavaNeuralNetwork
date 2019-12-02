package utilities;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;
import neuralnetwork.NetworkInput;
import org.junit.Test;

public class NetworkUtilitiesTest {

	@Test
	public void importFromInputStream() throws IOException {

		Stream<String> stream = Files.lines(Paths.get(
			"/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/java/neuralnetwork/mnist-in-csv/mnist_train.csv"));

		List<NetworkInput> inputs = NetworkUtilities.importFromInputStream(
			"/Users/edwincarlsson/Documents/Programmering/Java/NeuralNetwork/src/main/java/neuralnetwork/mnist-in-csv/mnist_train.csv",
			32, 0);
	}
}