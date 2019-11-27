package math.activations;

import java.util.Arrays;
import matrix.Matrix;
import org.junit.Before;
import org.junit.Test;

public class SoftmaxFunctionTest {

	private static int size = 10;

	Matrix[] randomMatrices = new Matrix[size];
	Matrix[] testMatrices = new Matrix[size / 2];

	@Before
	public void setUp() throws Exception {
		for (int i = 0; i < size; i++) {
			randomMatrices[i] = Matrix.randomFromRange(3, 1, -3, 3);
		}

		testMatrices[0] = Matrix.fromArray(new double[]{0.1, 0.2, 0.91});
		testMatrices[1] = Matrix.fromArray(new double[]{-0.1, -0.2, -0.91});
		testMatrices[2] = Matrix.fromArray(new double[]{1, 2, 10});
		testMatrices[3] = Matrix.fromArray(new double[]{-2, -5, -10});
		testMatrices[4] = Matrix.fromArray(new double[]{0.11111, 0.11111, 0.11111});


	}


	@Test
	public void applyFunction() {
		Matrix[] output = new Matrix[size + (size / 2)];
		SoftmaxFunction function = new SoftmaxFunction();
		int i = 0;

		for (Matrix m : randomMatrices) {
			output[i++] = function.applyFunction(m, null);
		}
		for (Matrix m : testMatrices) {
			output[i++] = function.applyFunction(m, null);
		}

		System.out.println(Arrays.toString(output));

	}

	@Test
	public void applyDerivative() {
	}
}