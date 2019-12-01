package math.activations;

import org.junit.Before;
import org.junit.Test;
import org.ujmp.core.DenseMatrix;

public class SoftmaxFunctionTest {

	private static int size = 10;

	DenseMatrix[] randomMatrices = new DenseMatrix[size];
	DenseMatrix[] testMatrices = new DenseMatrix[size / 2];

	@Before
	public void setUp() throws Exception {
		for (int i = 0; i < size; i++) {
			randomMatrices[i] = org.ujmp.core.Matrix.Factory.randn(3, 1);
		}

		testMatrices[0] = org.ujmp.core.Matrix.Factory
			.importFromArray(new double[][]{{0.1}, {0.2}, {0.91}});
		testMatrices[1] = org.ujmp.core.Matrix.Factory
			.importFromArray(new double[][]{{-0.1}, {-0.2}, {-0.91}});
		testMatrices[2] = org.ujmp.core.Matrix.Factory
			.importFromArray(new double[][]{{1}, {2}, {10}});
		testMatrices[3] = org.ujmp.core.Matrix.Factory
			.importFromArray(new double[][]{{-2}, {-5}, {-10}});
		testMatrices[4] = org.ujmp.core.Matrix.Factory
			.importFromArray(new double[][]{{0.11111}, {0.11111}, {0.11111}});


	}


	@Test
	public void applyFunction() {
		DenseMatrix[] output = new DenseMatrix[size + (size / 2)];
		SoftmaxFunction function = new SoftmaxFunction();
		int i = 0;

		for (DenseMatrix m : randomMatrices) {
			output[i++] = function.applyFunction(m);
		}
		for (DenseMatrix m : testMatrices) {
			output[i++] = function.applyFunction(m);
		}

	}

	@Test
	public void applyDerivative() {
	}
}