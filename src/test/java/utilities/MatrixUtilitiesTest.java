package utilities;

import static org.junit.Assert.assertEquals;

import math.activations.ActivationFunction;
import math.activations.TanhFunction;
import org.junit.Test;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class MatrixUtilitiesTest {

	ActivationFunction f = new TanhFunction();

	@Test
	public void map() {

		DenseMatrix matrix = Matrix.Factory.randn(7, 1);
		matrix.Matrix a = new matrix.Matrix(matrix);

		DenseMatrix mapped = MatrixUtilities.map(matrix, Math::tanh);
		matrix.Matrix mappedMatrix = a.map(Math::tanh);

		DenseMatrix derivative = f.applyDerivative(mappedMatrix);
		System.out.println(derivative);

		assertEquals(mapped.getValueSum(), mappedMatrix.getValueSum(), 10e-8);
	}

	@Test
	public void argMax() {

		DenseMatrix matrix = Matrix.Factory
			.importFromArray(new double[][]{{-0.11}, {0}, {-1110}, {2.312321312}});
		System.out.println(matrix);

		int out = MatrixUtilities.argMax(matrix);
		System.out.println(out);
	}
}