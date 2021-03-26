package reinforcement.env.space;

import java.security.SecureRandom;
import lombok.Getter;
import reinforcement.utils.Shape;

public abstract class Space<T> {

	@Getter
	protected final Shape shape;

	protected SecureRandom random;

	public Space(int... shapes) {
		if (shapes.length == 0) {
			this.shape = new Shape(0, 0, 0);
		} else if (shapes.length == 1) {
			this.shape = new Shape(shapes[0], 0, 0);
		} else if (shapes.length == 2) {
			this.shape = new Shape(shapes[0], shapes[1], 0);
		} else if (shapes.length == 3) {
			this.shape = new Shape(shapes[0], shapes[1], shapes[2]);
		} else {
			throw new IllegalArgumentException(
				"You need to specify some shapes, it is allowed the be an empty int[].");
		}

		this.random = new SecureRandom();
	}

	public void seed(long seed) {
		if (this.random == null) {
			this.random = new SecureRandom();
		}
		this.random.setSeed(seed);
	}

	public Shape shape() {
		return this.shape;
	}

	public abstract boolean contains(T value);

	public abstract T sample();

}
