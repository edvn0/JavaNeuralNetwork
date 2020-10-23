package math.costfunctions;

import static org.junit.Assert.assertEquals;

import java.util.List;
import math.activations.SoftmaxFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.ojalgo.matrix.Primitive64Matrix;

public class CrossEntropyCostFunctionTest {

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void calculateCostFunction() {
		CrossEntropyCostFunction<Primitive64Matrix> test = new CrossEntropyCostFunction<>();
		var data = List
			.of(new NetworkInput<Primitive64Matrix>(
					new OjAlgoMatrix(new double[][]{{0.25}, {0.25}, {0.25}, {0.25}}),
					new OjAlgoMatrix(new double[][]{{1}, {0}, {0}, {0}})),
				new NetworkInput<Primitive64Matrix>(
					new OjAlgoMatrix(new double[][]{{0.01}, {0.01}, {0.01}, {0.97}}),
					new OjAlgoMatrix(new double[][]{{0}, {0}, {0}, {1}})));

		double loss = test.calculateCostFunction(data);
		assertEquals(0.7083767843022996, loss, 1e-5);
	}
}