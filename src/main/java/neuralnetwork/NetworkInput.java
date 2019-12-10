package neuralnetwork;

import java.io.Serializable;
import org.ujmp.core.DenseMatrix;

/**
 * A wrapper to contain a input to a Neural Network, with both the label and the
 * data.
 */
public class NetworkInput implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -8743845031383184256L;
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
		return "NetworkInput{" + "data=" + data + ", label=" + label + '}';
	}
}
