package utilities;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class MathUtilitiesTest {

	@Test
	public void simpleMap() {
	}

	@Test
	public void argMaxHappy() {
		double[] vals = {0.1, 0.2, -1000, 1000, 0};
		assertEquals(3, MathUtilities.argMax(vals));
		assertEquals(1000, MathUtilities.max(vals), 1e-9);
	}

	@Test
	public void argMaxUnhappy() {
		double[] vals = {1, 2, 3, 4, 5, 1000};
		double[] vals0 = {1000, 1, 2, 3, 4, 5};
		assertEquals(vals.length - 1, MathUtilities.argMax(vals));
		assertEquals(0, MathUtilities.argMax(vals0));
	}
}