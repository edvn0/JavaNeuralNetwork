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
}
