package math.activations;

import java.util.concurrent.ThreadLocalRandom;
import org.junit.Before;
import org.junit.Test;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class TanhFunctionTest {

	private DenseMatrix in;
	private DenseMatrix div;
	private TanhFunction function;

	@Before
	public void init() {
		double[][] vec = new double[30][1];
		double[][] divVec = new double[30][1];
		function = new TanhFunction();
		for (int i = 0; i < vec.length; i++) {
			vec[i][0] = (ThreadLocalRandom.current().nextDouble() * 2) - 1;
			divVec[i][0] = Double.NEGATIVE_INFINITY;
		}

		in = Matrix.Factory.importFromArray(vec);
		div = Matrix.Factory.importFromArray(divVec);
	}

	@Test
	public void applyFunction() {
		DenseMatrix out1 = function.applyFunction(in);
		DenseMatrix out2 = function.applyFunction(div);
		System.out.println(out1);
		System.out.println(out2);
	}

	@Test
	public void applyDerivative() {
		in = function.applyFunction(in);
		div = function.applyFunction(div);
		DenseMatrix derivIn = function.applyDerivative(in, null);
		DenseMatrix derivDiv = function.applyDerivative(div, null);
		System.out.println(derivIn);
		System.out.println(derivDiv);
	}
}