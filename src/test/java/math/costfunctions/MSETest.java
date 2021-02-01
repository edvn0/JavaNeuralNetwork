package math.costfunctions;

import static org.junit.Assert.assertEquals;

import java.util.List;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;
import org.junit.Test;
import org.ojalgo.matrix.Primitive64Matrix;

public class MSETest {

	@Test
	public void testMse() {
		var mse = new MeanSquaredCostFunction<Primitive64Matrix>();
		var data = List.of(
			new NetworkInput<Primitive64Matrix>(
				new OjAlgoMatrix(new double[][]{{0.25}, {0.25}, {0.25}, {0.25}}),
				new OjAlgoMatrix(new double[][]{{1}, {0}, {0}, {0}})),
			new NetworkInput<Primitive64Matrix>(
				new OjAlgoMatrix(new double[][]{{0.01}, {0.01}, {0.01}, {0.97}}),
				new OjAlgoMatrix(new double[][]{{0}, {0}, {0}, {1}})));

		double trueValue = ((1 - 0.25) * (1 - 0.25) + 3 * (0.25 * 0.25))
			+ ((0.01 * 0.01 * 3) + (1 - 0.97) * (1 - 0.97));
		trueValue = trueValue / 2;

		double out = mse.calculateCostFunction(data);

		assertEquals(trueValue, out, 1e-7);
	}

	@Test
	public void testDerivative() {
		var mse = new MeanSquaredCostFunction<Primitive64Matrix>();
		var o = new NetworkInput<>(
			new OjAlgoMatrix(new double[][]{{0.25}, {0.2}, {0.2}, {0.2}}),
			new OjAlgoMatrix(new double[][]{{1}, {0}, {0}, {0}}));

		double[][] d = {{0.25 - 1}, {0.2}, {0.2}, {0.2}};

		assertEquals(new OjAlgoMatrix(d).multiply(2),
			mse.applyCostFunctionGradient(o.getData(), o.getLabel()));
	}

}
