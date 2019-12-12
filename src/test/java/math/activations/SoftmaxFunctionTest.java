package math.activations;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class SoftmaxFunctionTest {

	private DenseMatrix sm;

	@Before
	public void setUp() throws Exception {
		double[][] d = new double[10][1];
		double[] vals = {0.1, 0.2, -0.88, 0.13, 0.19, 0.001, 0.002, 0.0130, 0.76, 1};
		for (int i = 0; i < d.length; i++) {
			d[i][0] = vals[i];
		}
		sm = Matrix.Factory.importFromArray(d);
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void applyFunction() {
		SoftmaxFunction function = new SoftmaxFunction();
		System.out.println(function.applyFunction(sm));
	}

	@Test
	public void applyDerivative() {
		SoftmaxFunction function = new SoftmaxFunction();
		sm = function.applyFunction(sm);
		System.out.println(sm);
		System.out.println(function.applyDerivative(sm, null));
	}
}