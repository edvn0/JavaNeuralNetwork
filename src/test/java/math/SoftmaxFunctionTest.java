package math;

import math.activations.ActivationFunction;
import math.activations.SoftmaxFunction;
import matrix.Matrix;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class SoftmaxFunctionTest {

	private Matrix a = Matrix
		.fromArray(new double[]{0.11, 0.13, 0.3, 0.91, 0.75, -0.13, -0.99991, -0.1, -0.7});

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void applyFunctionTest() {
		ActivationFunction function = new SoftmaxFunction();
		Matrix output = function.applyFunction(a, null);
		System.out.println(output);
	}

	@Test
	public void applyDerivative() {
		ActivationFunction function = new SoftmaxFunction();
		Matrix output = function.applyFunction(a, null);
		output = function.applyDerivative(output, null);
		System.out.println(output);
	}

	@Test
	public void getName() {
	}

	@Test
	public void testToString() {
	}
}