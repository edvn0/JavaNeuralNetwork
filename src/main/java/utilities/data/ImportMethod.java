package utilities.data;

import java.io.IOException;
import java.util.List;
import neuralnetwork.NetworkInput;

public interface ImportMethod {

	List<NetworkInput> parseData(String in) throws IOException;

	void configureOptions(double[]... in);

}
