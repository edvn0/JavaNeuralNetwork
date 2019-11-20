package math;

import matrix.Matrix;
import org.junit.Test;

public class SoftMaxErrorFunctionTest {

	SoftMaxErrorFunction function = new SoftMaxErrorFunction();
	SoftmaxFunction functionSM = new SoftmaxFunction();

	Matrix input = Matrix
		.fromArray(new double[]{1, 0, 1, 0.1, -0.19, -0.23, 0.18, 1, 0.119332111});

	Matrix target = Matrix.fromArray(new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0});

	@Test
	public void applyErrorFunctionTest() {
		Matrix f = functionSM.applyFunction(input);
		System.out.println(f);
		Matrix fm = function.applyErrorFunction(input, target);
		System.out.println(fm);
	}
}