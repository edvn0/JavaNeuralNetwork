package reinforcement.env.space;

import reinforcement.utils.Shape;

public class Box1D extends Space<Double> {

	private final double low;
	private final double high;

	public Box1D(double low, double high, int... shape) {
		super(shape);
		this.low = low;
		this.high = high;
	}

	@Override
	public Shape shape() {
		return this.shape;
	}

	@Override
	public boolean contains(final Double value) {
		return value >= low && value <= high;
	}

	@Override
	public Double sample() {
		return low + (high - low) * this.random.nextDouble();
	}
}
