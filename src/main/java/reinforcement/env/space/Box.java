package reinforcement.env.space;

public class Box extends Space<Double> {

	private double[][] limits;

	public Box(double[][] limits) {
		super(limits.length);
		this.limits = limits;
	}

	@Override
	public boolean contains(final Double value) {
		return false;
	}

	@Override
	public Double sample() {
		return null;
	}
}
