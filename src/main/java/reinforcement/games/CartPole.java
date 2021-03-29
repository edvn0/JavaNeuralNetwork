package reinforcement.games;

import reinforcement.env.BaseEnvironment;
import reinforcement.env.space.Box;
import reinforcement.env.space.Discrete;
import reinforcement.utils.EnvInfo;
import reinforcement.utils.EnvObservation;
import reinforcement.utils.Sars;

public class CartPole extends BaseEnvironment<Integer, Double> {

	private double x;
	private double xDot;
	private double theta;
	private double thetaDot;

	public CartPole() {
		this.actionSpace = new Discrete(2);
		this.observationSpace = new Box(
			new double[][]{{-4.8, 4.8}, {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY},
				{-0.418, 0.418}, {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY}});
		this.x = random.nextDouble() * 0.05;
		this.xDot = random.nextDouble() * 0.05;
		this.theta = random.nextDouble() * 0.05;
		this.thetaDot = random.nextDouble() * 0.05;
	}

	@Override
	public Sars step(final Integer a) {
		if (!this.actionSpace.contains(a)) {
			EnvInfo info = new EnvInfo();
			info.addInfo("Invalid action.");
			return new Sars(this.observe(), 0, true, info);
		}

		double forceMag = 10.0;
		var force = a == 1 ? forceMag : -1 * forceMag;
		var cosTheta = Math.cos(theta);
		var sinTheta = Math.sin(theta);

		double massPole = 0.1;
		double length = 0.5;
		double poleMassLength = massPole * length;
		double massCart = 1.0;
		double totalMass = massCart + massPole;
		var temp =
			(force * poleMassLength * Math.pow(thetaDot, 2) * sinTheta) / totalMass;
		double gravity = 9.8;
		var thetaAcc = (gravity * sinTheta - cosTheta * temp) / (length * (4 / 3
			- massPole * Math.pow(cosTheta, 2) / totalMass));
		var xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass;

		double tau = 0.02;
		x = x + tau * xDot;
		xDot = xDot + tau * xAcc;
		theta = theta + tau * thetaDot;
		thetaDot = thetaDot + tau * thetaAcc;

		var state = new EnvObservation(x, xDot, theta, thetaDot);

		double thetaThresholdRadians = 12 * 2 * Math.PI / 360;
		double xThreshold = 2.4;
		var done =
			x < -xThreshold || x > xThreshold || theta < -thetaThresholdRadians
				|| theta > thetaThresholdRadians;

		double reward = 0.0d;
		if (!done) {
			reward = 1;
		}

		return new Sars(state, reward, done, null);
	}

	@Override
	public Sars reset() {
		var x = random.nextDouble() * 0.05;
		var xDot = random.nextDouble() * 0.05;
		var theta = random.nextDouble() * 0.05;
		var thetaDot = random.nextDouble() * 0.05;
		this.x = x;
		this.xDot = xDot;
		this.theta = theta;
		this.thetaDot = thetaDot;
		return new Sars(new EnvObservation(x, xDot, theta, thetaDot), 0, false, null);
	}

	@Override
	public void seed(final long seed) {

	}

	@Override
	public boolean didWin() {
		return false;
	}

	@Override
	public void close() throws Exception {

	}

	@Override
	protected EnvObservation observe() {
		return new EnvObservation(this.x, this.xDot, this.theta, this.thetaDot);
	}
}
