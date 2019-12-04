package neuralnetwork;

import org.ujmp.core.DenseMatrix;

public class NetworkInput {

	private DenseMatrix data;
	private DenseMatrix label;

	public NetworkInput(DenseMatrix data, DenseMatrix label) {
		this.data = data;
		this.label = label;
	}

	public DenseMatrix getData() {
		return data;
	}

	public DenseMatrix getLabel() {
		return label;
	}

	@Override
	public String toString() {
		return "NetworkInput{" + "data=" + data
			+ ", label=" + label
			+ '}';
	}
}
