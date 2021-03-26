package reinforcement.env.space;

import java.security.SecureRandom;
import reinforcement.utils.Shape;

public class Discrete extends Space<Integer> {

	private final int spaceSize;

	public Discrete(final int dim1) {
		super(dim1);
		if (this.random == null) {
			this.random = new SecureRandom();
		}
		this.spaceSize = dim1;
	}

	@Override
	public Shape shape() {
		return this.shape;
	}

	@Override
	public boolean contains(final Integer value) {
		return value <= spaceSize && value >= 0;
	}

	@Override
	public Integer sample() {
		return this.random.nextInt(this.spaceSize);
	}
}
