package reinforcement.env.space;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;
import reinforcement.utils.Shape;

public class Box1DTest {

	Box1D box;

	@Before
	public void setup() {
		this.box = new Box1D(-1, 1, 1);
	}


	@Test
	public void shape() {
		assertEquals(this.box.shape, new Shape(1, 0, 0));
	}

	@Test
	public void contains() {
	}

	@Test
	public void sampleAndBounds() {
		double sample = this.box.sample();

		boolean inBounds = this.box.contains(sample);

		assertTrue(inBounds);
	}
}