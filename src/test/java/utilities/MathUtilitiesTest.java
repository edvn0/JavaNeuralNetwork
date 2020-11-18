package utilities;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class MathUtilitiesTest {

	@Test
	public void simpleMap() {
	}

	@Test
	public void argMax() {
		double[] vals = {0.1, 0.2, -1000, 1000,
			0}; // test big negatives, small positives and big positives
		assertEquals(3, MathUtilities.argMax(vals));
	}
}