package neuralnetwork.inputs;

import math.linearalgebra.Matrix;

/**
 * A wrapper to contain a input to a Neural Network, with both the label and the data.
 */
public class NetworkInput<M> {

	private final Matrix<M> data;
	private final Matrix<M> label;

	public NetworkInput(Matrix<M> data, Matrix<M> label) {
		this.data = data;
		this.label = label;
	}

	public Matrix<M> getData() {
		return data;
	}

	public Matrix<M> getLabel() {
		return label;
	}

	@Override
	public String toString() {
		return "NetworkInput{" + "data=" + data + ", label=" + label + '}';
	}
}
