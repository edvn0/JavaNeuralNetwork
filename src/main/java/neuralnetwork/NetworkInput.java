package neuralnetwork;

import matrix.Matrix;

public class NetworkInput {

	private Matrix data;
	private Matrix label;

	public NetworkInput(Matrix data, Matrix label) {
		this.data = data;
		this.label = label;
	}

	public Matrix getData() {
		return data;
	}

	public Matrix getLabel() {
		return label;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("NetworkInput{");
		sb.append("data=").append(data);
		sb.append(", label=").append(label);
		sb.append('}');
		return sb.toString();
	}
}
