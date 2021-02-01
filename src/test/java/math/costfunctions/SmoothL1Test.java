package math.costfunctions;

import static org.junit.Assert.assertEquals;

import java.util.List;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.inputs.NetworkInput;
import org.junit.Test;
import org.ojalgo.matrix.Primitive64Matrix;

public class SmoothL1Test {

	@Test
	public void smoothL1Test() {
		SmoothL1CostFunction<Primitive64Matrix> l1 = new SmoothL1CostFunction<>();
		l1.setL1(5);

		var data = List.of(new NetworkInput<Primitive64Matrix>(
			new OjAlgoMatrix(new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
			new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463}, {17.218656},
				{36.586389}, {87.288984}})));

		assertEquals(9.719187818293003, l1.calculateCostFunction(data), 1e-7);
	}

	@Test
	public void smoothL1TestIteration() {
		SmoothL1CostFunction<Primitive64Matrix> l1 = new SmoothL1CostFunction<>();
		l1.setL1(5);

		var data = List.of(new NetworkInput<Primitive64Matrix>(
				new OjAlgoMatrix(new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463}, {17.218656},
					{36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})),
			new NetworkInput<Primitive64Matrix>(new OjAlgoMatrix(
				new double[][]{{24.0}, {50.0}, {15d}, {38d}, {87d}}),
				new OjAlgoMatrix(new double[][]{{21.549452}, {47.464463},
					{17.218656}, {36.586389}, {87.288984}})));

		assertEquals(9.719187818293003, l1.calculateCostFunction(data), 1e-7);
	}

}
