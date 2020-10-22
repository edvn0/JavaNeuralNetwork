package math.linearalgebra;

import java.util.function.Function;
import utilities.exceptions.MatrixException;

public interface Matrix<M> {

	/**
	 * Cols or this Matrix<M>
	 *
	 * @return columns
	 */
	int rows();

	/**
	 * Rows of this Matrix<M>
	 *
	 * @return rows
	 */
	int cols();

	/**
	 * Matrix<M> multiplication, should throw if cols and rows do not match.
	 * Contract is This X in, i.e. this_rows*this_cols X in_cols*in_rows
	 *
	 * @param otherMatrix right operand
	 *
	 * @return new Matrix<M> multiplied
	 */
	Matrix<M> multiply(Matrix<M> otherMatrix);

	/**
	 * Element wise multiplication of two matrices.
	 *
	 * @param otherMatrix right operand
	 *
	 * @return new element wise multiplied Matrix<M>
	 */
	Matrix<M> hadamard(Matrix<M> otherMatrix);

	/**
	 * Multiply each element with this scalar
	 *
	 * @param scalar to multiply with
	 *
	 * @return scaled with scalar
	 */
	Matrix<M> multiply(double scalar);

	/**
	 * Add in to this Matrix<M>
	 *
	 * @param in right operand
	 *
	 * @return this + in
	 */
	Matrix<M> add(Matrix<M> in);

	/**
	 * Add in to all elements of this.
	 *
	 * @param in scalar operand
	 *
	 * @return this.map(e - > e + in)
	 */
	Matrix<M> add(double in);

	/**
	 * Subtract in from all elements of this
	 *
	 * @param in scalar operand
	 *
	 * @return this.map(e - > e - in);
	 */
	Matrix<M> subtract(double in);

	/**
	 * Substract in from this Matrix<M>
	 *
	 * @param in right operand
	 *
	 * @return this[i][j] -= in[i][j]
	 */
	Matrix<M> subtract(Matrix<M> in);

	/**
	 * Divide all elements by in
	 *
	 * @param in scalar operand
	 *
	 * @return in.map(e - > e / in);
	 */
	Matrix<M> divide(double in);

	/**
	 * Map this Matrix<M> to a double, useful for reduce or trace implementations
	 *
	 * @param mapping f: This -> double
	 *
	 * @return a double value
	 */
	double map(Function<Matrix<M>, Double> mapping);

	/**
	 * Map each element with this function
	 *
	 * @param mapping f: Double -> Double each element
	 *
	 * @return this.map(e - > mapping ( e));
	 */
	Matrix<M> mapElements(Function<Double, Double> mapping);

	default void mapElementsMutable(Function<Double, Double> mapping) {
		setDelegate(this.mapElements(mapping).delegate());
	}

	/**
	 * Sum this Matrix<M> over all entries.
	 *
	 * @return sum of this
	 */
	double sum();

	/**
	 * Max of this Matrix<M> over all entries.
	 *
	 * @return max of this
	 */
	double max();

	/**
	 * Index along a column of max, should only be used for vectors.
	 *
	 * @return index of max
	 */
	int argMax();

	/**
	 * The delegate part of the delegate pattern.
	 */
	M delegate();

	void setDelegate(M delegate);

	/**
	 * Transpose this Matrix<M>.
	 *
	 * @return transpose.
	 */
	Matrix<M> transpose();

	/**
	 * Divide each element by corresponding element in other matrix.
	 *
	 * @param right other matrix.
	 *
	 * @return this /= other.
	 */
	Matrix<M> divide(Matrix<M> right);

	/**
	 * @return A vector with all values equal to max.
	 */
	Matrix<M> maxVector();

	/**
	 * A matrix with all zeroes.
	 *
	 * @param rows M
	 * @param cols N
	 *
	 * @return M X N (0...0)
	 */
	Matrix<M> zeroes(int rows, int cols);

	/**
	 * A matrix with all ones.
	 *
	 * @param rows M
	 * @param cols N
	 *
	 * @return M X N (0...0)
	 */
	Matrix<M> ones(int rows, int cols);

	/**
	 * Identity matrix, ones in diagonal.
	 *
	 * @param rows M
	 * @param cols N
	 *
	 * @return M X N (0...0)
	 */
	Matrix<M> identity(int rows, int cols);

	/**
	 * Representation of the underlying delegate.
	 */
	String toString();

	enum MatrixType {
		VECTOR, SQUARE, ZEROES, ONES, IDENTITY
	}

	/**
	 * Norm of a vector.
	 *
	 * @return square root of sum of squared components.
	 *
	 * @throws MatrixException if cols() != 1.
	 */
	double norm() throws MatrixException;

	/**
	 * Name of underlying implementation.
	 */
	String name();

	double[][] rawCopy();

	/**
	 * Returns AA = A^2 for square matrix.
	 *
	 * @return A^2
	 *
	 * @throws MatrixException if not square matrix, cols == rows.
	 */
	default Matrix<M> squareMatrix() throws MatrixException {

		if (cols() != rows()) {
			throw new MatrixException("This was not a square matrix.");
		}

		return this.multiply(this);
	}
}
