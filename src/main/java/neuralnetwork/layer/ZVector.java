package neuralnetwork.layer;

import java.util.StringJoiner;
import math.linearalgebra.Matrix;
import utilities.exceptions.MatrixException;

public class ZVector<M> {

	private Matrix<M> zVector;

	public ZVector(Matrix<M> in) {

		if (in.cols() != 1) {
			throw new MatrixException(String.format(
				"You need to supply a vector to this constructor, supplied was a matrix of %d X %d size.",
				in.cols(), in.rows()));
		}

		this.zVector = in;
	}

	public ZVector(ZVector<M> in) {
		this(in.zVector);
	}

	public Matrix<M> getZVector() {
		return zVector;
	}

	public void setZVector(Matrix<M> zVector) {
		this.zVector = zVector;
	}

	@Override
	public String toString() {
		return new StringJoiner(", ", ZVector.class.getSimpleName() + "[", "]")
			.add("zVector=" + zVector)
			.toString();
	}
}