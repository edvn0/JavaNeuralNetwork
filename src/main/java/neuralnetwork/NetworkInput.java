package neuralnetwork;

import java.io.Serializable;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

/**
 * A wrapper to contain a input to a Neural Network, with both the label and the
 * data.
 */
public class NetworkInput implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = -8743845031383184256L;
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
		return "NetworkInput{" + "data=" + data + ", label=" + label + '}';
	}
}
