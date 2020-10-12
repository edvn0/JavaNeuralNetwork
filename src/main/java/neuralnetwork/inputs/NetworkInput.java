package neuralnetwork.inputs;

import math.linearalgebra.Matrix;
import neuralnetwork.layer.ZVector;

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

	public static <M> NetworkInput<M> of(final ZVector<M> a, final ZVector<M> b) {
		return new NetworkInput<>(a.getZVector(), b.getZVector());
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
