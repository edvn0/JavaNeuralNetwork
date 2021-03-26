package reinforcement.env.space;

public class RewardRange {

	private final double min;
	private final double max;

	public RewardRange(double min, double max) {
		this.min = min;
		this.max = max;
	}

	public boolean contains(Double d) {
		return d >= min && d <= max;
	}

}
