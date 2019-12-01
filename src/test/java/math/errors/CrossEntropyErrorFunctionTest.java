package math.errors;

import java.util.ArrayList;
import java.util.List;
import neuralnetwork.NetworkInput;
import org.junit.Test;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class CrossEntropyErrorFunctionTest {

	DenseMatrix matrix = Matrix.Factory
		.importFromArray(new double[][]{{0.1}, {0.1}, {0.7}, {0.05}, {0.05}});
	DenseMatrix corr = Matrix.Factory.importFromArray(new double[][]{{0}, {0}, {1}, {0}, {0}});

	@Test
	public void calculateCostFunction() {
		List<NetworkInput> list = new ArrayList<>();
		list.add(new NetworkInput(matrix, corr));
		CrossEntropyErrorFunction entropyErrorFunction = new CrossEntropyErrorFunction();
		System.out.println(entropyErrorFunction.calculateCostFunction(list));
	}
}