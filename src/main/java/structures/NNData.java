package structures;

public class NNData {

	private double[] inputs;
	private double[] targets;

	public NNData(double[] inputs, double[] targets) {
		this.inputs = inputs;
		this.targets = targets;
	}

	public double[] getInputs() {
		return inputs;
	}

	public double[] getTargets() {
		return targets;
	}
}
