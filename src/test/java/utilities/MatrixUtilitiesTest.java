package utilities;

import matrix.Matrix;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class MatrixUtilitiesTest {

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

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void networkOutputsSoftMax() {
		Matrix[] out = new Matrix[size + (size / 2)];

		int k = 0;
		for (Matrix m : randomMatrices) {
			System.out.println("Matrix index: " + k);
			out[k++] = MatrixUtilities.networkOutputsSoftMax(m);
			System.out.println();
		}

		for (Matrix m : testMatrices) {
			System.out.println("Matrix index: " + k);
			out[k++] = MatrixUtilities.networkOutputsSoftMax(m);
			System.out.println();
		}

		for (Matrix o : out) {
			System.out.println(o);
		}


	}

	@Test
	public void networkOutputsMax() {
	}
}