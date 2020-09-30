package math.linearalgebra;

import java.util.function.Function;

public abstract class Matrix<M> {

    public Matrix<M> squareMatrix() {
        return this.multiply(this);
    }

    /**
     * Cols or this matrix
     *
     * @return columns
     */
    public abstract int rows();

    /**
     * Rows of this matrix
     *
     * @return rows
     */
    public abstract int cols();

    /**
     * Matrix multiplication, should throw if cols and rows do not match. Contract
     * is This X in, i.e. this_rows*this_cols X in_cols*in_rows
     *
     * @param otherMatrix right operand
     * @return new matrix multiplied
     */
    public abstract Matrix<M> multiply(Matrix<M> otherMatrix);

    public abstract Matrix<M> hadamard(Matrix<M> otherMatrix);

    /**
     * Multiply each element with this scalar
     *
     * @param scalar to multiply with
     * @return scaled with scalar
     */
    public abstract Matrix<M> multiply(double scalar);

    /**
     * Add in to this matrix
     *
     * @param in right operand
     * @return this + in
     */
    public abstract Matrix<M> add(Matrix<M> in);

    /**
     * Add in to all elements of this.
     *
     * @param in scalar operand
     * @return this.map(e - > e + in)
     */
    public abstract Matrix<M> add(double in);

    /**
     * Subtract in from all elements of this
     *
     * @param in scalar operand
     * @return this.map(e - > e - in);
     */
    public abstract Matrix<M> subtract(double in);

    /**
     * Substract in from this matrix
     *
     * @param in right operand
     * @return this[i][j] -= in[i][j]
     */
    public abstract Matrix<M> subtract(Matrix<M> in);

    /**
     * Divide all elements by in
     *
     * @param in scalar operand
     * @return in.map(e - > e / in);
     */
    public abstract Matrix<M> divide(double in);

    /**
     * Map this matrix to a double, useful for reduce or trace implementations
     *
     * @param mapping f: This -> double
     * @return a double value
     */
    public abstract double map(Function<Matrix<M>, Double> mapping);

    /**
     * Map each element with this function
     *
     * @param mapping f: Double -> Double each element
     * @return this.map(e - > mapping ( e));
     */
    public abstract Matrix<M> mapElements(Function<Double, Double> mapping);

    /**
     * Sum this matrix over all entries.
     *
     * @return sum of this
     */
    public abstract double sum();

    /**
     * Max of this matrix over all entries.
     *
     * @return max of this
     */
    public abstract double max();

    /**
     * Index along a column of max, should only be used for vectors.
     *
     * @return index of max
     */
    public abstract int argMax();

    /**
     * Transpose this matrix.
     *
     * @return transpose.
     */
    public abstract Matrix<M> transpose();

    public abstract M delegate();

    public abstract Matrix<M> divide(Matrix<M> right);

    public abstract Matrix<M> maxVector();

    public abstract Matrix<M> zeroes(int rows, int cols);

    public abstract Matrix<M> ones(int rows, int cols);

    public abstract Matrix<M> identity(int rows, int cols);

    public abstract String toString();

    public enum MatrixType {
        VECTOR, SQUARE, ZEROES, ONES, IDENTITY
    }

    public abstract double square();

    public abstract String name();
}
