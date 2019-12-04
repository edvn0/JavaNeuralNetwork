package neuralnetwork;

import java.io.Serializable;
import org.ujmp.core.DenseMatrix;

public interface Trainable extends Serializable {

	/**
	 * Takes some values to classify, and compares to the correct classification. This method only
	 * takes 1 case.
	 *
	 * @param inputs a {@link NetworkInput} object to wrap two {@link DenseMatrix} objects (data and
	 *               labels.
	 */
	void train(NetworkInput inputs);

	/**
	 * Predicts a classification from a {@link DenseMatrix} input.
	 *
	 *
	 * Example usage: {@link org.ujmp.core.Matrix}(double[][]{{0},{1}}) becomes {@link
	 * org.ujmp.core.Matrix}(double[][]{{0.987}}) for the XOR problem.
	 *
	 * @param in values to be predicted.
	 *
	 * @return A classification of input values.
	 */
	DenseMatrix predict(DenseMatrix in);

	String toString();

}
