package math.linearalgebra;

import java.util.function.Function;

public interface Matrix<M> {

    default Matrix<M> squareMatrix() {
        return this.multiply(this);
    }

    /**
     * Cols or this matrix
     *
     * @return columns
     */
    int rows();

    /**
     * Rows of this matrix
     *
     * @return rows
     */
    int cols();

    /**
     * Matrix multiplication, should throw if cols and rows do not match.
     * Contract is This X in, i.e. this_rows*this_cols X in_cols*in_rows
     *
     * @param otherMatrix right operand
     * @return new matrix multiplied
     */
    Matrix<M> multiply(Matrix<M> otherMatrix);

    Matrix<M> hadamard(Matrix<M> otherMatrix);
    

    /**
     * Multiply each element with this scalar
     *
     * @param scalar to multiply with
     * @return scaled with scalar
     */
    Matrix<M> multiply(double scalar);

    /**
     * Add in to this matrix
     *
     * @param in right operand
     * @return this + in
     */
    Matrix<M> add(Matrix<M> in);

    /**
     * Add in to all elements of this.
     *
     * @param in scalar operand
     * @return this.map(e - > e + in)
     */
    Matrix<M> add(double in);

    /**
     * Subtract in from all elements of this
     *
     * @param in scalar operand
     * @return this.map(e - > e - in);
     */
    Matrix<M> subtract(double in);

    /**
     * Substract in from this matrix
     *
     * @param in right operand
     * @return this[i][j] -= in[i][j]
     */
    Matrix<M> subtract(Matrix<M> in);

    /**
     * Divide all elements by in
     *
     * @param in scalar operand
     * @return in.map(e - > e / in);
     */
    Matrix<M> divide(double in);

    /**
     * Map this matrix to a double, useful for reduce or trace implementations
     *
     * @param mapping f: This -> double
     * @return a double value
     */
    double map(Function<Matrix<M>, Double> mapping);

    /**
     * Map each element with this function
     *
     * @param mapping f: Double -> Double each element
     * @return this.map(e - > mapping ( e));
     */
    Matrix<M> mapElements(Function<Double, Double> mapping);

    /**
     * Sum this matrix over all entries.
     *
     * @return sum of this
     */
    double sum();

    /**
     * Max of this matrix over all entries.
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
     * Transpose this matrix.
     *
     * @return transpose.
     */
    Matrix<M> transpose();

    M delegate();

    Matrix<M> divide(Matrix<M> right);

    Matrix<M> maxVector();

    Matrix<M> zeroes(int rows, int cols);

    Matrix<M> ones(int rows, int cols);

    Matrix<M> identity(int rows, int cols);

    String toString();

    enum MatrixType {
        VECTOR, SQUARE, ZEROES, ONES, IDENTITY
    }

	double square();
}
