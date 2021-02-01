package math.linearalgebra.simple;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.apache.log4j.BasicConfigurator;
import org.junit.Before;
import org.junit.Test;

public class SimpleMatrixTest {

	@Before
	public void setUp() throws Exception {
		BasicConfigurator.configure();
	}

	@Test
	public void rows() {
		SimpleMatrix matrix = new SimpleMatrix(new double[][]{{1, 1, 2}, {1, 3, 1}});
		assertEquals(2, matrix.rows());
	}

	@Test
	public void cols() {
		SimpleMatrix matrix = new SimpleMatrix(new double[][]{{1, 1, 2}, {1, 3, 1}});
		assertEquals(3, matrix.cols());
	}

	@Test
	public void multiply() {
		SimpleMatrix id = new SimpleMatrix(new double[][]{{1, 0}, {0, 1}});
		SimpleMatrix out = new SimpleMatrix(new double[][]{{2, 3}, {1, 5}});
		SimpleMatrix expectedIdOut = new SimpleMatrix(out);

		SimpleMatrix matrix = new SimpleMatrix(new double[][]{{1, 1}, {0, 1}});
		SimpleMatrix otherMatrix = new SimpleMatrix(new double[][]{{2, 3}, {1, 5}});
		SimpleMatrix expectedMult = new SimpleMatrix(new double[][]{{3, 8}, {1, 5}});

		System.out.println(out.multiply(id));

		assertEquals(expectedIdOut, out.multiply(id));
		assertEquals(expectedMult, matrix.multiply(otherMatrix));
	}

	@Test
	public void testMultiply() {

		SimpleMatrix m1 = new SimpleMatrix(new double[]{1, 5, -3, 5});

		SimpleMatrix out1Matrix = m1.multiply(-2);
		SimpleMatrix out2Matrix = m1.multiply(0.01);

		assertEquals(new SimpleMatrix(new double[]{-2, -10, 6, -10}), out1Matrix);
		assertEquals(new SimpleMatrix(new double[]{0.01 * 1, 0.01 * 5, 0.01 * -3, 0.01 * 5}),
			out2Matrix);
	}

	@Test
	public void add() {
	}

	@Test
	public void testAdd() {
	}

	@Test
	public void subtract() {
	}

	@Test
	public void divide() {
	}

	@Test
	public void map() {
	}

	@Test
	public void mapElements() {
		SimpleMatrix m = new SimpleMatrix(
			new double[][]{{9, 1_000_000, 4}, {1, 16, 49}, {25, 81, 100}});
		assertEquals(new SimpleMatrix(new double[][]{{3, 1000, 2}, {1, 4, 7}, {5, 9, 10}}),
			m.mapElements(Math::sqrt));
	}

	@Test
	public void mutableMapTest() {
		SimpleMatrix m = new SimpleMatrix(
			new double[][]{{9, 1_000_000, 4}, {1, 16, 49}, {25, 81, 100}});
		m.mapElementsMutable(Math::sqrt);

		assertEquals(new SimpleMatrix(new double[][]{{3, 1000, 2}, {1, 4, 7}, {5, 9, 10}}), m);
	}

	@Test
	public void hadamard() {
	}

	@Test
	public void sum() {
	}

	@Test
	public void max() {
	}

	@Test
	public void argMax() {
	}

	@Test
	public void delegate() {
	}

	@Test
	public void setDelegate() {
	}

	@Test
	public void transpose() {
	}

	@Test
	public void maxVector() {
	}

	@Test
	public void zeroes() {
	}

	@Test
	public void ones() {
	}

	@Test
	public void identity() {
	}

	@Test
	public void norm() {
	}

	@Test
	public void name() {
	}

	@Test
	public void rawCopy() {
	}

	@Test
	public void copy() {
		SimpleMatrix m = new SimpleMatrix(new double[][]{{1}, {2}, {3}});
		SimpleMatrix copy = (SimpleMatrix) m.copy();
		assertEquals(copy, m);
		assertNotEquals(m.hashCode(), copy.hashCode());
	}

	@Test
	public void testToString() {
	}

	@Test
	public void testEquals() {
	}
}