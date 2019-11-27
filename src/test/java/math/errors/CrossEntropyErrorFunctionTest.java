package math.errors;

import java.util.ArrayList;
import java.util.List;
import matrix.Matrix;
import neuralnetwork.NetworkInput;
import org.junit.Before;
import org.junit.Test;

public class CrossEntropyErrorFunctionTest {

	List<NetworkInput> list = new ArrayList<>();

	private static int size = 10;

	Matrix[] randomMatrices = new Matrix[size];
	Matrix[] testMatrices = new Matrix[size / 2];

	@Before
	public void setUp() throws Exception {
		for (int i = 0; i < size; i++) {
			randomMatrices[i] = Matrix.randomFromRange(3, 1, -3, 3);
			list.add(new NetworkInput(randomMatrices[i], getCorrectLabel(randomMatrices[i])));
		}

		testMatrices[0] = Matrix.fromArray(new double[]{0.1, 0.2, 0.91});
		testMatrices[1] = Matrix.fromArray(new double[]{-0.1, -0.2, -0.91});
		testMatrices[2] = Matrix.fromArray(new double[]{1, 2, 10});
		testMatrices[3] = Matrix.fromArray(new double[]{-2, -5, -10});
		testMatrices[4] = Matrix.fromArray(new double[]{0.11111, 0.6, 0.11111});

		list.add(new NetworkInput(testMatrices[0], Matrix.fromArray(new double[]{0, 0, 1})));
		list.add(new NetworkInput(testMatrices[0], Matrix.fromArray(new double[]{1, 0, 0})));
		list.add(new NetworkInput(testMatrices[0], Matrix.fromArray(new double[]{0, 0, 1})));
		list.add(new NetworkInput(testMatrices[0], Matrix.fromArray(new double[]{1, 0, 0})));
		list.add(new NetworkInput(testMatrices[0], Matrix.fromArray(new double[]{0, 1, 0})));

	}

	private Matrix getCorrectLabel(final Matrix randomMatrix) {
		double[] data = randomMatrix.toArray();
		double max = data[0];
		int index = 0;
		for (int i = 1; i < data.length; i++) {
			if (data[i] > max) {
				index = i;
				max = data[i];
			}
		}

		double[] out = new double[data.length];
		out[index] = 1;

		return Matrix.fromArray(out);
	}

	@Test
	public void applyErrorFunction() {
		CrossEntropyErrorFunction entropyErrorFunction = new CrossEntropyErrorFunction();
		entropyErrorFunction.calculateCostFunction(list);

	}
}