package math;

import matrix.Matrix;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class SoftmaxFunctionTest {

	private Matrix a = Matrix
		.fromArray(new double[]{0.11, 0.13, 1, 0.91, 0.75, -0.13, -0.99991, -0.1, -0.7});

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void applyFunctionTest() {
		ActivationFunction function = new SoftmaxFunction();
		Matrix output =function.applyFunction(a);
		System.out.println(output);
	}

	@Test
	public void applyDerivative() {
	}

	@Test
	public void getName() {
	}

	@Test
	public void testToString() {
	}
}