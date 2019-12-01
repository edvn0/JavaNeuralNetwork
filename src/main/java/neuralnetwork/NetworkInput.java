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
		final StringBuilder sb = new StringBuilder("NetworkInput{");
		sb.append("data=").append(data);
		sb.append(", label=").append(label);
		sb.append('}');
		return sb.toString();
	}
}
