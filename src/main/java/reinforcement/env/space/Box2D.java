package reinforcement.env.space;

public class Box2D extends Space<Double> {

	private final double[] rangeX;
	private final double[] rangeY;

	public Box2D(double[] rangeX, double[] rangeY) {
		super(rangeX.length, rangeY.length);
		this.rangeX = rangeX;
		this.rangeY = rangeY;
	}

	// TODO: fix this
	@Override
	public boolean contains(final Double value) {
		return (value >= rangeX[0] && value <= rangeX[1]) && (value >= rangeY[0]
			&& value <= rangeY[1]);
	}

	@Override
	public Double sample() {
		return null;
	}
}
