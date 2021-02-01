package math.activations;

import static org.junit.Assert.assertEquals;

import math.linearalgebra.Matrix;
import math.linearalgebra.simple.SMatrix;
import math.linearalgebra.simple.SimpleMatrix;
import org.junit.Test;

public class LinearActivationTest {

	@Test
	public void testLinear() {
		ActivationFunction<SMatrix> s = new LinearFunction<>(1);
		double[][] vars = {{1}, {2}, {3}};
		SimpleMatrix m = new SimpleMatrix(vars);

		Matrix<SMatrix> exc = s.function(m);

		assertEquals("Linear function should just do nothing", m.delegate(), exc.delegate());
	}

	@Test
	public void testLinearDerivative() {
		ActivationFunction<SMatrix> s = new LinearFunction<>(1);
		double[][] vars = {{1}, {2}, {3}};
		SimpleMatrix m = new SimpleMatrix(vars);

		Matrix<SMatrix> exc = s.derivative(m);

		double[][] deriv = {{1}, {1}, {1}};
		SimpleMatrix t = new SimpleMatrix(deriv);

		assertEquals("Linear function should just do nothing", t, exc);
	}

}
